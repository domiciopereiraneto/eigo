"""
This script contains the Eigo class, which implements an optimization engine for generating images using Stable Diffusion XL (SDXL) 
guided by aesthetic and prompt alignment scores. The engine supports two optimization methods: CMA-ES (standard, sep-CMA-ES and VD-CMA) and Adam.
"""

import sys
import os
import shutil
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from diffusers import StableDiffusionXLPipeline
import random
from PIL import Image
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import csv
from pptx import Presentation
from pptx.util import Inches
import cma
from cma.restricted_gaussian_sampler import GaussVDSampler 
from datasets import load_dataset
import clip
import argparse

class Eigo:
    def __init__(self, config_parameters):
        self.parameters = config_parameters

        if config_parameters["predictor"] == 0:
            predictor_name = 'simulacra'
        elif config_parameters["predictor"] == 1:
            predictor_name = 'laionv1'
        elif config_parameters["predictor"] == 2:
            predictor_name = 'laionv2'
        else:
            raise ValueError("Invalid predictor option.")

        if config_parameters["optimization_method"] == "adam":
            method_save_name = "adam"
        elif config_parameters["optimization_method"] == "cmaes":
            if config_parameters["cmaes_variant"] == "cmaes":
                method_save_name = "cmaes"
            elif config_parameters["cmaes_variant"] == "sep":
                method_save_name = "sepcmaes"
            elif config_parameters["cmaes_variant"] == "vd":
                method_save_name = "vdcmae"
            else:
                raise ValueError(f"Unknown CMA-ES variant: {config_parameters['cmaes_variant']}")
        else:
            raise ValueError(f"Unknown optimization method: {config_parameters['optimization_method']}")

        self.OUTPUT_FOLDER = f"{config_parameters['results_folder']}/{method_save_name}_clip_{predictor_name}_sdxlturbo_{config_parameters['seed']}_a{int(config_parameters['alpha']*100)}_b{int(config_parameters['beta']*100)}"

        # Save the selected prompts and their categories to a text file in the results folder
        os.makedirs(self.OUTPUT_FOLDER, exist_ok=True)

        # Check if a GPU is available and if not, use the CPU
        self.device = "cuda:" + str(config_parameters["cuda"]) if torch.cuda.is_available() else "cpu"

        # Load the SDXL pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            config_parameters["model_id"],
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        self.call_with_grad = self.pipe.__class__.__call__.__wrapped__.__get__(self.pipe, self.pipe.__class__)

        clip_model_name = "ViT-L/14"  # CLIP model name
        self.clip_model, self.clip_preprocess = clip.load(clip_model_name, device=self.device)

        # Initialize the aesthetic model
        if config_parameters["predictor"] == 0:
            from aesthetic_evaluation.src import simulacra_rank_image
            self.aesthetic_model = simulacra_rank_image.SimulacraAesthetic(self.device)
            self.model_name = 'SAM'
        elif config_parameters["predictor"] == 1:       
            from aesthetic_evaluation.src import laion_rank_image     
            self.aesthetic_model = laion_rank_image.LAIONAesthetic(self.device, clip_model=clip_model_name)
            self.model_name = 'LAIONV1'
        elif config_parameters["predictor"] == 2:
            from aesthetic_evaluation.src import laion_v2_rank_image
            self.aesthetic_model = laion_v2_rank_image.LAIONV2Aesthetic(self.device, clip_model=clip_model_name)
            self.model_name = 'LAIONV2'
        else:
            raise ValueError("Invalid predictor option.")

        # Differentiable CLIP score evaluation for Adam
        self._CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1)
        self._CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1)
        
    def generate_image_from_embeddings_cmaes(self, prompt_embeds, pooled_prompt_embeds, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        out = self.pipe(
            prompt_embeds              = prompt_embeds,
            pooled_prompt_embeds       = pooled_prompt_embeds,
            guidance_scale             = 0.0,
            num_inference_steps        = self.parameters["num_inference_steps"],
            generator                  = generator,
            height                     = self.parameters["height"],
            width                      = self.parameters["width"],
            output_type               = "pt"
        )["images"]

        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)      # HWC
        return image.to(self.device)

    def generate_image_from_embeddings_adam(self, text_embeddings, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        prompt_embeds = text_embeddings[0]
        pooled_prompt_embeds = text_embeddings[1]

        out = self.call_with_grad(
            prompt_embeds              = prompt_embeds,
            pooled_prompt_embeds       = pooled_prompt_embeds,
            guidance_scale             = 0.0,
            num_inference_steps        = self.parameters["num_inference_steps"],
            generator                  = generator,
            height                     = self.parameters["height"],
            width                      = self.parameters["width"],
            output_type               = "pt"
        )["images"]

        image = out.clamp(0, 1).squeeze(0).permute(1, 2, 0)      # HWC
        return image.to(self.device)

    def aesthetic_evaluation(self, image):
        # image is a tensor of shape [H, W, C]
        # Convert to [N, C, H, W] and ensure it's in float32
        image_input = image.permute(2, 0, 1).to(torch.float32)  # [1, C, H, W]

        if self.parameters["predictor"] == 0:
            # Simulacra Aesthetic Model
            score = self.aesthetic_model.predict_from_tensor(image_input)
        elif self.parameters["predictor"] == 1 or self.parameters["predictor"] == 2:
            # LAION Aesthetic Predictor V1 and V2
            score = self.aesthetic_model.predict_from_tensor(image_input)
        else:
            return torch.tensor(0.0, device=self.device)

        return score

    def evaluate_clip_score_cmaes(self, image_tensor, prompt):
        # Convert the image tensor to a PIL image
        image = (image_tensor * 255).clamp(0, 255).byte()
        image = Image.fromarray(image.cpu().numpy())

        # Preprocess the image
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Tokenize the prompt
        text_input = clip.tokenize([prompt]).to(self.device)

        # Compute the CLIP embeddings
        image_features = self.clip_model.encode_image(image_input)
        text_features = self.clip_model.encode_text(text_input)

        # Normalize the features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute the cosine similarity (CLIP score)
        clip_score = (image_features @ text_features.T)

        return clip_score

    def evaluate_clip_score_adam(self, image_tensor, text_features):
        self.clip_model.eval()

        # --- differentiable preprocess (no PIL, no .byte) ---
        img = image_tensor.permute(2,0,1).unsqueeze(0)            # [1,C,H,W]
        img = img.to(device=self.device, dtype=torch.float32)
        img = F.interpolate(img, size=(224,224), mode="bicubic", align_corners=False)
        mean = self._CLIP_MEAN.to(img.device, img.dtype)
        std  = self._CLIP_STD.to(img.device, img.dtype)
        img = (img - mean) / std

        # Encode image WITH grad (through CLIP image tower)
        image_features = self.clip_model.encode_image(img).float()
        image_features = F.normalize(image_features, dim=-1, eps=1e-6)

        sim = (image_features @ text_features.T).squeeze()  # scalar
        return sim

    def format_time(self, seconds):
        seconds = int(seconds)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def evaluate(self, input_embedding, seed, embedding_shape, selected_prompt, save_path=None):
        # x is a NumPy array representing the embedding vector
        # Convert it to a torch tensor

        # Reshape the embedding to the original shape
        split = np.prod(embedding_shape[0])
        pe  = torch.tensor(input_embedding[:split],  dtype=torch.float32, device=self.device).view(embedding_shape[0])
        ppe = torch.tensor(input_embedding[split:], dtype=torch.float32, device=self.device).view(embedding_shape[1])

        with torch.no_grad():
            image = self.generate_image_from_embeddings_cmaes(pe, ppe, seed)

            aesthetic_score = self.aesthetic_evaluation(image).item()
            clip_score = self.evaluate_clip_score_cmaes(image, selected_prompt).item()
        # CMA-ES minimizes the function, so we need to invert the score if higher is better

        fitness_1 = self.parameters["alpha"]*aesthetic_score/self.parameters["max_aesthetic_score"]
        fitness_2 = self.parameters["beta"]*clip_score/self.parameters["max_clip_score"]

        fitness = fitness_1 + fitness_2

        if save_path is not None:
            # Save the generated image
            image_np = image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(save_path)

        return -fitness, aesthetic_score, clip_score, fitness_1, fitness_2 

    def run_cmaes_optimization(self):

        def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
            lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
            upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

            plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_label is not None:
                plt.xlabel(x_label)

        def save_plot_results(results, results_folder):
            # Plot main fitness evolution
            plt.figure(figsize=(10, 6))  # Increase figure size
            plot_mean_std(results['generation'], results['avg_fitness'], results['std_fitness'], "Fitness")
            plt.plot(results['generation'], results['max_fitness'], 'r-', label="Best Fitness")
            plt.ylim(0, 1.1)
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/fitness_evolution.png")
            plt.close()

            # Plot aesthetic score evolution
            plt.figure(figsize=(10, 6))  # Increase figure size
            plot_mean_std(results['generation'], results['avg_aesthetic_score'], results['std_aesthetic_score'], "Population")
            plt.plot(results['generation'], results['max_aesthetic_score'], 'r-', label="Best")
            plt.ylim(0, 10)
            plt.xlabel('Generation')
            plt.ylabel('Aesthetic Score')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/aesthetic_score_evolution.png")
            plt.close()

            # Plot clip score evolution
            plt.figure(figsize=(10, 6))  # Increase figure size
            plot_mean_std(results['generation'], results['avg_clip_score'], results['std_clip_score'], "Population")
            plt.plot(results['generation'], results['max_clip_score'], 'r-', label="Best")
            plt.ylim(0, 0.6)
            plt.xlabel('Generation')
            plt.ylabel('CLIP Score')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/clip_score_evolution.png") 
            plt.close()   

        seed = self.parameters["seed"]
        selected_prompt = self.parameters["selected_prompt"]

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"
        os.makedirs(results_folder, exist_ok=True)

        with torch.no_grad():
            (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                    selected_prompt,
                    negative_prompt="",                 
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
            )

        # Set CMA-ES options
        es_options = {
            'seed': seed,
            'popsize': self.parameters["pop_size"],
            'maxiter': self.parameters["num_generations"],
            'verb_filenameprefix': results_folder + '/outcmaes',  # Save logs
            'verb_log': 0,  # Disable log output
            'verbose': -9,  # Suppress console output
        }

        if self.parameters["cmaes_variant"] == "cmaes":
            print("Using standard CMA-ES")
        elif self.parameters["cmaes_variant"] == "sep":
            print("Using sep-CMA-ES")
            es_options['CMA_diagonal'] = True
        elif self.parameters["cmaes_variant"] == "vd":
            print("Using VD-CMA-ES")
            es_options = GaussVDSampler.extend_cma_options(es_options)
        else:
            raise ValueError(f"Unknown CMA-ES variant: {self.parameters['cmaes_variant']}")

        trainable_params_init = torch.cat([
            prompt_embeds.flatten(),
            pooled_prompt_embeds.flatten()
            ]).cpu().numpy()

        sh_prompt  = prompt_embeds.shape         
        sh_pooled  = pooled_prompt_embeds.shape  

        text_embeddings_init_shape = [sh_prompt, sh_pooled]

        es = cma.CMAEvolutionStrategy(trainable_params_init, self.parameters["sigma"], es_options)

        with torch.no_grad():
            initial_image = self.generate_image_from_embeddings_cmaes(prompt_embeds.clone(), pooled_prompt_embeds.clone(), seed)
            image_np = initial_image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_0.png")

            initial_fitness, initial_aesthetic_score, initial_clip_score, initial_fitness_1, initial_fitness_2 = self.evaluate(trainable_params_init, seed, text_embeddings_init_shape, selected_prompt)

        time_list = [0]
        best_aesthetic_score_overall = initial_aesthetic_score
        best_clip_score_overall = initial_clip_score
        best_fitness_overall = initial_fitness
        best_text_embeddings_overall = trainable_params_init

        start_time = time.time()
        generation = 0

        max_fit_list = [-initial_fitness]
        avg_fit_list = [-initial_fitness]
        std_fit_list = [0]

        max_aesthetic_score_list = [initial_aesthetic_score]
        avg_aesthetic_score_list = [initial_aesthetic_score]
        std_aesthetic_score_list = [0]

        max_clip_score_list = [initial_clip_score]
        avg_clip_score_list = [initial_clip_score]
        std_clip_score_list = [0]

        max_fitness_1_list = [initial_fitness_1]
        avg_fitness_1_list = [initial_fitness_1]
        std_fitness_1_list = [0]

        max_fitness_2_list = [initial_fitness_2]
        avg_fitness_2_list = [initial_fitness_2]
        std_fitness_2_list = [0]

        while not es.stop():
            print(f"Generation {generation+1}/{self.parameters['num_generations']}")

            os.makedirs(results_folder+"/gen_%d" % (generation+1), exist_ok=True)

            # Ask for new candidate solutions
            solutions = es.ask()
            # Evaluate candidate solutions
            tmp_fitnesses = []
            tmp_fitness_1 = []
            tmp_fitness_2 = []
            aesthetic_scores = []
            clip_scores = []

            ind_id = 1
            for x in solutions:
                save_path = results_folder + "/gen_%d/id_%d.png" % (generation+1, ind_id)
                fitness, aesthetic_score, clip_score, fitness_1, fitness_2 = self.evaluate(x, seed, text_embeddings_init_shape, selected_prompt, save_path)
                tmp_fitnesses.append(fitness)
                tmp_fitness_1.append(fitness_1)
                tmp_fitness_2.append(fitness_2)
                aesthetic_scores.append(aesthetic_score)
                clip_scores.append(clip_score)
                ind_id += 1
            # Tell CMA-ES the fitnesses
            es.tell(solutions, tmp_fitnesses)

            # Record statistics
            fitnesses = [-f for f in tmp_fitnesses]  # Convert back to positive scores

            max_fit = max(fitnesses)
            avg_fit = np.mean(fitnesses)
            std_fit = np.std(fitnesses)

            max_fit_list.append(max_fit)
            avg_fit_list.append(avg_fit)
            std_fit_list.append(std_fit)

            max_aesthetic_score = max(aesthetic_scores)
            avg_aesthetic_score = np.mean(aesthetic_scores)
            std_aesthetic_score = np.std(aesthetic_scores)

            max_aesthetic_score_list.append(max_aesthetic_score)
            avg_aesthetic_score_list.append(avg_aesthetic_score)
            std_aesthetic_score_list.append(std_aesthetic_score)

            max_clip_score = max(clip_scores)
            avg_clip_score = np.mean(clip_scores)
            std_clip_score = np.std(clip_scores)

            max_clip_score_list.append(max_clip_score)
            avg_clip_score_list.append(avg_clip_score)
            std_clip_score_list.append(std_clip_score)

            max_fitness_1 = max(tmp_fitness_1)
            avg_fitness_1 = np.mean(tmp_fitness_1)
            std_fitness_1 = np.std(tmp_fitness_1)

            max_fitness_1_list.append(max_fitness_1)
            avg_fitness_1_list.append(avg_fitness_1)
            std_fitness_1_list.append(std_fitness_1)

            max_fitness_2 = max(tmp_fitness_2)
            avg_fitness_2 = np.mean(tmp_fitness_2)
            std_fitness_2 = np.std(tmp_fitness_2)

            max_fitness_2_list.append(max_fitness_2)
            avg_fitness_2_list.append(avg_fitness_2)
            std_fitness_2_list.append(std_fitness_2)

            # Get best solution so far
            best_x = es.result.xbest
            best_fitness = -es.result.fbest  # Convert back to positive score

            with torch.no_grad():
                # Generate and save the best image
                split = np.prod(text_embeddings_init_shape[0])
                best_pe  = torch.tensor(best_x[:split],  dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
                best_ppe = torch.tensor(best_x[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
                best_image = self.generate_image_from_embeddings_cmaes(best_pe, best_ppe, seed)
                image_np = best_image.detach().clone().cpu().numpy()
                image_np = (image_np * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np)
                pil_image.save(results_folder + "/best_%d.png" % (generation+1))

            if best_fitness > best_fitness_overall:
                best_fitness_overall = best_fitness
                best_text_embeddings_overall = best_x

            generation += 1

            elapsed_time = time.time() - start_time
            generations_done = generation
            generations_left = self.parameters["num_generations"] - generations_done
            average_time_per_generation = elapsed_time / generations_done
            estimated_time_remaining = average_time_per_generation * generations_left

            formatted_time_remaining = self.format_time(estimated_time_remaining)

            time_list.append(elapsed_time)

            # Save the metrics
            results = pd.DataFrame({
                "generation": list(range(0, generation + 1)),
                "prompt": [selected_prompt] + [''] * generation,
                "avg_fitness": avg_fit_list,
                "std_fitness": std_fit_list,
                "max_fitness": max_fit_list,
                "avg_aesthetic_score": avg_aesthetic_score_list,
                "std_aesthetic_score": std_aesthetic_score_list,
                "max_aesthetic_score": max_aesthetic_score_list,
                "avg_clip_score": avg_clip_score_list,
                "std_clip_score": std_clip_score_list,
                "max_clip_score": max_clip_score_list,
                "elapsed_time": time_list
            })

            results.to_csv(f"{results_folder}/fitness_results.csv", index=False, na_rep='nan')

            save_plot_results(results, results_folder)

            # Print stats
            print(f"Generation {generation}/{self.parameters['num_generations']}: Max fitness: {max_fit}, Avg fitness: {avg_fit}, Max aesthetic score: {max_aesthetic_score}, Avg aesthetic score: {avg_aesthetic_score}, Max clip score: {max_clip_score}, Avg clip score: {avg_clip_score}, Estimated time remaining: {formatted_time_remaining}")

        # Save the overall best image
        with torch.no_grad():
            split = np.prod(text_embeddings_init_shape[0])
            best_overall_pe  = torch.tensor(best_text_embeddings_overall[:split],  dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[0])
            best_overall_ppe = torch.tensor(best_text_embeddings_overall[split:], dtype=torch.float32, device=self.device).view(text_embeddings_init_shape[1])
            best_image = self.generate_image_from_embeddings_cmaes(best_overall_pe, best_overall_ppe, seed)
        best_image_np = best_image.detach().cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_all.png")

        return results_folder

    def run_adam_optimization(self):

        def plot_results(results, results_folder):
            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['aesthetic_score'], label="Aesthetic Score")
            plt.xlabel('Iteration')
            plt.ylabel('Aesthetic Score')
            plt.title('Aesthetic Score Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/aesthetic_evolution.png")
            plt.close()

            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['clip_score'], label="CLIP Score")
            plt.xlabel('Iteration')
            plt.ylabel('CLIP Score')
            plt.title('CLIP Score Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/clip_evolution.png")
            plt.close()

            # Plot all losses in one plot
            plt.figure(figsize=(10, 6))  # Increase figure size
            plt.plot(results['iteration'], results['combined_loss'], label="Combined Loss")
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Loss Evolution')
            plt.grid()
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))  # Move legend outside the plot
            plt.tight_layout()  # Adjust layout
            plt.savefig(results_folder + "/loss_evolution.png")
            plt.close()

        def plot_mean_std(x_axis, m_vec, std_vec, description, title=None, y_label=None, x_label=None):
            lower_bound = [M_new - Sigma for M_new, Sigma in zip(m_vec, std_vec)]
            upper_bound = [M_new + Sigma for M_new, Sigma in zip(m_vec, std_vec)]

            plt.plot(x_axis, m_vec, '--', label=description + " Avg.")
            plt.fill_between(x_axis, lower_bound, upper_bound, alpha=.3, label=description + " Avg. ± SD")
            if title is not None:
                plt.title(title)
            if y_label is not None:
                plt.ylabel(y_label)
            if x_label is not None:
                plt.xlabel(x_label)
    
        seed = self.parameters["seed"]
        selected_prompt = self.parameters["selected_prompt"]

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        print(f"Selected prompt: {selected_prompt}")

        results_folder = f"{self.OUTPUT_FOLDER}/results_{self.model_name}_{seed}"
        os.makedirs(results_folder, exist_ok=True)

        # Text features don't depend on your params; compute w/o grad
        with torch.no_grad():
            text_tokens = clip.tokenize([selected_prompt]).to(self.device)
            text_features = self.clip_model.encode_text(text_tokens).float()
            text_features = F.normalize(text_features, dim=-1, eps=1e-6)

        (prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
                    selected_prompt,
                    negative_prompt="",                 
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False
        )
        text_embeddings_init = [prompt_embeds.detach().clone(), pooled_prompt_embeds.detach().clone()]
        text_embeddings = [torch.nn.Parameter(prompt_embeds.clone()), torch.nn.Parameter(pooled_prompt_embeds.clone())]

        with torch.no_grad():
            initial_image = self.generate_image_from_embeddings_adam(text_embeddings_init, seed)
            image_np = initial_image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_0.png")

        aesthetic_score = self.aesthetic_evaluation(initial_image)

        clip_score = self.evaluate_clip_score_adam(initial_image, text_features)

        initial_combined_score = self.parameters["alpha"] * aesthetic_score / self.parameters["max_aesthetic_score"] + self.parameters["beta"] * clip_score / self.parameters["max_clip_score"]
        initial_combined_loss = 1 - initial_combined_score

        combined_score_list = [initial_combined_score.item()]
        combined_loss_list = [initial_combined_loss.item()]
        time_list = [0]
        best_score = initial_combined_score
        best_text_embeddings = text_embeddings_init.copy()

        optimizer = torch.optim.Adam(text_embeddings, lr=self.parameters["adam_lr"], betas=(self.parameters["adam_beta1"],self.parameters["adam_beta2"]), weight_decay=self.parameters["adam_weight_decay"], eps=self.parameters["adam_eps"])  

        start_time = time.time()

        # Add lists to store the metrics
        aesthetic_score_list = [aesthetic_score.item()]
        clip_score_list = [clip_score.item()]

        for iteration in range(1, self.parameters["num_iterations"] + 1):
            print(f"Iteration {iteration}/{self.parameters['num_iterations']}")

            optimizer.zero_grad()

            #with torch.autocast(device_type=device, dtype=torch.float16):
            image = self.generate_image_from_embeddings_adam(text_embeddings, seed)
            aesthetic_score = self.aesthetic_evaluation(image)
            clip_score = self.evaluate_clip_score_adam(image, text_features)
            combined_score = self.parameters["alpha"] * aesthetic_score / self.parameters["max_aesthetic_score"] + self.parameters["beta"] * clip_score / self.parameters["max_clip_score"]
            combined_loss = 1 - combined_score

            # Calculate gradients
            combined_loss.backward()
            # Update parameters
            optimizer.step()

            # Append metrics to their respective lists
            aesthetic_score_list.append(aesthetic_score.item())
            clip_score_list.append(clip_score.item())

            if combined_score.item() > best_score:
                best_score = combined_score.item()
                best_text_embeddings = text_embeddings.copy()

            combined_score_list.append(combined_score.item())
            combined_loss_list.append(combined_loss.item())

            image_np = image.detach().clone().cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_np)
            pil_image.save(f"{results_folder}/it_{iteration}.png")

            elapsed_time = time.time() - start_time
            iterations_done = iteration
            iterations_left = self.parameters["num_iterations"] - iteration
            average_time_per_iteration = elapsed_time / iterations_done
            estimated_time_remaining = average_time_per_iteration * iterations_left

            formatted_time_remaining = self.format_time(estimated_time_remaining)

            time_list.append(elapsed_time)

            # Save metrics to the results DataFrame
            results = pd.DataFrame({
                "iteration": list(range(0, iteration + 1)),
                "prompt": [selected_prompt] + [''] * iteration,
                "combined_score": combined_score_list,
                "combined_loss": combined_loss_list,
                "aesthetic_score": aesthetic_score_list,
                "clip_score": clip_score_list,
                "elapsed_time": time_list
            })

            results.to_csv(f"{results_folder}/score_results.csv", index=False, na_rep='nan')

            # Plot and save the fitness evolution
            plot_results(results, results_folder)

            # Print stats
            print(f"Iteration {iteration}/{self.parameters['num_iterations']}: Combined Score: {combined_score.item()}, Aesthetic Score: {aesthetic_score.item()}, CLIP Score: {clip_score.item()}, Estimated time remaining: {formatted_time_remaining}")

        # Save the overall best image
        with torch.no_grad():
            best_image = self.generate_image_from_embeddings_adam(best_text_embeddings, seed)
        best_image_np = best_image.detach().cpu().numpy()
        best_image_np = (best_image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(best_image_np)
        pil_image.save(f"{results_folder}/best_all.png")

        return results_folder
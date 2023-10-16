from typing import List, Dict, Tuple, Any, Union, Optional
from library.webuiapi import WebUIApi, QueuedTaskResult, ControlNetUnit, raw_b64_img
import os
import time
from PIL import Image
from threading import Thread

# image open functions
def open_controlnet_image(path:str):
    # open controlnet images, TODO : open json file as image (openpose)
    control_ref_img = Image.open(path)
    control_ref_img = control_ref_img.convert("RGB")
    return control_ref_img

def open_mask_image(path:str):
    mask_ref_img = Image.open(path)
    mask_ref_img = mask_ref_img.convert("RGB")
    return mask_ref_img

# helper functions
def wait_for_result(result:QueuedTaskResult, result_list:list,
                    check_interval:int=1):
    """wait for result to finish"""
    # function for thread
    while not result.is_finished():
        time.sleep(check_interval)
    result_list.append(result)

def concat_horizontally(*images):
    # get max height
    # get max width
    # create new image
    max_height = max([img.size[1] for img in images])
    total_width = sum([img.size[0] for img in images])
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.size[0]
    return new_img

def concat_vertically(*images):
    # get max height
    # get max width
    # create new image
    max_width = max([img.size[0] for img in images])
    total_height = sum([img.size[1] for img in images])
    new_img = Image.new('RGB', (max_width, total_height))
    y_offset = 0
    for img in images:
        new_img.paste(img, (0, y_offset))
        y_offset += img.size[1]
    return new_img

def generate_animated_gif(images:List[Image.Image], filename:str, duration:int=100):
    # generate animated gif
    images[0].save(filename, save_all=True, append_images=images[1:], duration=duration, loop=0)

class InstanceHolder:
    def __init__(self, instance:WebUIApi):
        self.instance = instance
        
    def upload_lora(self, lora_dir:str, path:str="test_by_api"):
        self.instance.set_overwrite(True) # set overwrite to True
        lora_name = os.path.basename(lora_dir).split(".")[0]
        self.instance.upload_lora(lora_dir, path) # upload lora model from target_path to server's test_lora dir
        return lora_name

    def process_test_latent_couple(self, control_ref_path:str, mask_ref_path:str, 
                    lora_name_1:str="haibara_dynamic_lora_reg2",
                    lora_name_2:str="haibara_compare_autotrain",
                    tag_1:str="character name is haibara ai",
                    tag_2:str="character name is haibara ai",
                    mode_1:str='NONE', # 'NONE', 'FACE', "KEEPCHAR"...
                    mode_2:str='NONE',
                    cnet_model:str="control_v11p_sd15_openpose [cab727d4]",
                    preprocessor = "none",
                    **image_generation_args
                    ):
        
        default_generation_args = {
            "sampler_name":"Euler a",
            'hr_second_pass_steps':20,
            'width':512,
            'height':768,
            'enable_hr':True,
            'hr_scale':2,
            'denoising_strength':0.45,
            'seed':1003,
        }
        apiInstance = self.instance
        default_generation_args.update(image_generation_args)
        control_ref_img = open_controlnet_image(control_ref_path)
        mask_ref_img = open_mask_image(mask_ref_path)
        control_unit = ControlNetUnit(input_image=control_ref_img, module=preprocessor, model=cnet_model)
        result_control_latent_couple = apiInstance.txt2img_task(
            prompt = fr""" night, sky, forest, detailed, street, city, high angle
        AND high angle shot,  {tag_1}, standing, masterpiece, detailed,high angle <lora:{lora_name_1}:1:lbw={mode_1}>
        AND {tag_2}, masterpiece, standing, beaufiful,high angle  <lora:{lora_name_2}:1:lbw={mode_2}>""",
            negative_prompt = "nsfw, easynegative, blurry, noised, 3 legs, weird eyes, wrong eyes",
            alwayson_scripts={
                'latent couple extension': {
                    "args" : [ # enabled,
                        'Mask',
                        # raw_divisions, raw_positions, raw_weights, raw_end_at_step, alpha_blend, *cur_weight_sliders, 
                        '1:1,1:2,1:2','0:0,0:0,0:1', '0.2,0.8,0.8', 150, 0.2, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
                        , #sketch at last
                        True, #denoise mask argument
                        raw_b64_img(mask_ref_img),
                    ]
                }
        ,
                #composable lora etc...
                "composable lora" : {
                    "args" : [
                        True, True, False
                    ]
                }
                },
            controlnet_units=[control_unit],
            **default_generation_args
        )
        return result_control_latent_couple
    
    
    def process_test_and_wait(self, control_ref_path:str, mask_ref_path:str, result_list:list, should_wait_finish:bool=True,
                            test_args:dict={}, target_func=process_test_latent_couple):
        """process test and wait for result to finish"""
        instance = self.instance
        result_control_latent_couple = target_func(instance, control_ref_path, mask_ref_path, **test_args)
        thread = Thread(target=wait_for_result, args=(result_control_latent_couple, result_list))
        thread.start()
        if should_wait_finish:
            thread.join()
        return result_control_latent_couple


    def test_division(self,
                prompt_1_arg:str = "1boy",
                prompt_2_arg:str = "1girl",
                lora_stop_steps = (10, 10),
                extra_args:dict = dict(),
                controlnet_path:str = "test_controlnet.png",
                controlnet_preprocessor:str = "none",
                controlnet_model:str = "control_v11p_sd15_openpose [cab727d4]",
                base_prompt:str = "masterpiece, night, line art, simple background, street",
                common_prompt:str = "(best-quality:0.8), detailed face, perfect anime illustration",
                negative_prompt:str = "(worst quality:0.8), verybadimagenegative_v1.3, easynegative, (surreal:0.8), (modernism:0.8), (art deco:0.8), (art nouveau:0.8)",
                regional_prompter_args:Optional[dict] = None,
            ) -> QueuedTaskResult:
        """
        Tests regional prompter with division mode.
        Arguments :
            prompt_1_arg : prompt for first region
            prompt_2_arg : prompt for second region
            lora_stop_steps : tuple of (stop_step, stop_step_hires)
            extra_args : extra arguments for txt2img_task
        
        Returns :
            QueuedTaskResult object
            (use QueuedTaskResult.get_image() to get image)
        """
        instance = self.instance
        result_container = []
        extra_args_base = {
            'enable_hr':True,
            'hr_upscaler':'R-ESRGAN 4x+ Anime6B',
            'width':768,
            'height':1152,
            'hr_scale':1.3,
            'denoising_strength':0.6,
            #'seed':1003,
        }
        regional_prompter_base_args = {
            "Division Mode" : "Columns",
            "Divide Ratio" : "1, 1",
            "Base Ratio" : "0.2",
            "LoRA in negative textencoder" : "0",
            "LoRA in negative U-net" : "0",
            "threshold" : "0",
            "Use Base Prompt" : False,
            "Use Common Prompt" : False,
            "Use Common Negative Prompt" : False,
            "Disable Convert 'AND' to 'BREAK'" : False,
        }
        regional_prompter_base_args.update(regional_prompter_args or dict())
        # check if other arguments are valid
        for key in regional_prompter_base_args:
            if key not in ["Division Mode", "Divide Ratio", "Base Ratio", "LoRA in negative textencoder",
                           "LoRA in negative U-net", "threshold", "Use Base Prompt", "Use Common Prompt",
                           "Use Common Negative Prompt", "Disable Convert 'AND' to 'BREAK'"]:
                raise ValueError(f"Invalid argument {key}")
        extra_args_base.update(extra_args)
        result = instance.txt2img_task(
            # assume 3-region image
            prompt=f"""{base_prompt} BREAK
            {common_prompt}, {prompt_2_arg} BREAK
            {common_prompt}, {prompt_1_arg} """,
            negative_prompt=negative_prompt,
            #prompt = "masterpiece, night BREAK red BREAK blue BREAK yellow",
            controlnet_units=[
                    ControlNetUnit(input_image=open_controlnet_image(controlnet_path),
                                module=controlnet_preprocessor, model=controlnet_model)
            ],
            alwayson_scripts={
                "regional prompter" : {
                    "args" : [
                        True, # Active
                        False, # debug
                        'Matrix', # can be Matrix, Mask, Prompt
                        regional_prompter_base_args["Division Mode"], # can be Horizontal, Vertical, Columns, Rows
                        "Mask", # not used
                        "Prompt", # not used
                        regional_prompter_base_args["Divide Ratio"], # Divide Ratio
                        regional_prompter_base_args["Base Ratio"], # Base Ratio
                        regional_prompter_base_args["Use Base Prompt"], # Use base prompt
                        regional_prompter_base_args["Use Common Prompt"], # Use common prompt
                        regional_prompter_base_args["Use Common Negative Prompt"], # Use common negative prompt
                        "Latent", # Attention, Latent
                        regional_prompter_base_args["Disable Convert 'AND' to 'BREAK'"], # disable convert 'AND' to 'BREAK'
                        regional_prompter_base_args["LoRA in negative textencoder"], # LoRA in negative textencoder
                        regional_prompter_base_args["LoRA in negative U-net"], # LoRA in negative U-net
                        regional_prompter_base_args["threshold"], # threshold
                        "", # divide mode does not need mask
                        str(lora_stop_steps[0]), # LoRA stop step
                        str(lora_stop_steps[1]), # LoRA Hires stop step
                        False, # flip "," and ";"
                    ]
                }
            },
            **extra_args_base
        )
        wait_for_result(result, result_container)
        return result_container[0]
        
    
    def test_mask(self,
                prompt_1_arg:str = "1boy",
                prompt_2_arg:str = "1girl",
                lora_stop_steps = (10, 10),
                extra_args:dict = dict(),
                mask_path:str = "test_mask.png",
                controlnet_path:str = "test_controlnet.png",
                controlnet_preprocessor:str = "none",
                controlnet_model:str = "control_v11p_sd15_openpose [cab727d4]",
                base_prompt:str = "masterpiece, night, line art, simple background, street",
                common_prompt:str = "(best-quality:0.8), detailed face, perfect anime illustration",
                negative_prompt:str = "(worst quality:0.8), verybadimagenegative_v1.3, easynegative, (surreal:0.8), (modernism:0.8), (art deco:0.8), (art nouveau:0.8)",
                regional_prompter_args:Optional[dict] = None,
            ):
        """
        Tests regional prompter with mask mode.
        Arguments :
            prompt_1_arg : prompt for first region
            prompt_2_arg : prompt for second region
            lora_stop_steps : tuple of (stop_step, stop_step_hires)
            extra_args : extra arguments for txt2img_task
            
        Returns :
            QueuedTaskResult object
            (use QueuedTaskResult.get_image() to get image)
        """
        instance = self.instance
        result_container = []
        extra_args_base = {
            'enable_hr':True,
            'hr_upscaler':'R-ESRGAN 4x+ Anime6B',
            'width':768,
            'height':1152,
            'hr_scale':1.3,
            'denoising_strength':0.6,
            #'seed':1003,
        }
        regional_prompter_base_args = {
            "Divide Ratio" : "1, 1",
            "Base Ratio" : "0.2",
            "LoRA in negative textencoder" : "0",
            "LoRA in negative U-net" : "0",
            "threshold" : "0",
            "Use Base Prompt" : False,
            "Use Common Prompt" : False,
            "Use Common Negative Prompt" : False,
            "Disable Convert 'AND' to 'BREAK'" : False,
        }
        regional_prompter_base_args.update(regional_prompter_args or dict())
        # check if other arguments are valid
        for key in regional_prompter_base_args:
            if key not in ["Division Mode", "Divide Ratio", "Base Ratio", "LoRA in negative textencoder",
                           "LoRA in negative U-net", "threshold", "Use Base Prompt", "Use Common Prompt",
                           "Use Common Negative Prompt", "Disable Convert 'AND' to 'BREAK'"]:
                raise ValueError(f"Invalid argument {key}")
        extra_args_base.update(extra_args)
        result = instance.txt2img_task(
            # assume 3-region image
            prompt=f"""{base_prompt} BREAK
            {common_prompt}, {prompt_2_arg} BREAK
            {common_prompt}, {prompt_1_arg} """,
            negative_prompt=negative_prompt,
            #prompt = "masterpiece, night BREAK red BREAK blue BREAK yellow",
            controlnet_units=[
                    ControlNetUnit(input_image=open_controlnet_image(controlnet_path),
                                module=controlnet_preprocessor, model=controlnet_model)
            ],
            alwayson_scripts={
                "regional prompter" : {
                    "args" : [
                        True, # Active
                        False, # debug
                        'Mask', # can be Matrix, Mask, Prompt
                        "Horizontal", # can be Horizontal, Vertical, Columns, Rows
                        "Mask", # not used
                        "Prompt", # not used
                        regional_prompter_base_args["Divide Ratio"], # Divide Ratio
                        regional_prompter_base_args["Base Ratio"], # Base Ratio
                        regional_prompter_base_args["Use Base Prompt"], # Use base prompt
                        regional_prompter_base_args["Use Common Prompt"], # Use common prompt
                        regional_prompter_base_args["Use Common Negative Prompt"], # Use common negative prompt
                        "Latent", # Attention, Latent
                        regional_prompter_base_args["Disable Convert 'AND' to 'BREAK'"], # disable convert 'AND' to 'BREAK'
                        regional_prompter_base_args["LoRA in negative textencoder"], # LoRA in negative textencoder
                        regional_prompter_base_args["LoRA in negative U-net"], # LoRA in negative U-net
                        regional_prompter_base_args["threshold"], # threshold
                        raw_b64_img(open_mask_image(mask_path)), # base64 encoded image or 'absolute path' if saved on server
                        str(lora_stop_steps[0]), # LoRA stop step
                        str(lora_stop_steps[1]), # LoRA Hires stop step
                        False, # flip "," and ";"
                    ]
                }
            },
            **extra_args_base
        )
        wait_for_result(result, result_container)
        return result_container[0]

    def test_and_save(self,
            args_1:List[str] = list(), args_2:List[str] = list(), suffix = "", filepath = './examples',
            mask_path:str = "",
            controlnet_path:str = "",
            prompts:dict = dict(),
            return_images:bool = False,
            extra_args:dict = dict(),
            regional_prompter_args:Optional[dict] = None,
            lora_stop_steps:Tuple[int] = (10, 10),
            test_dict:Dict[str, Any] = dict(),
        ):
        """
        Tests mask generation and save result to filepath
        """
        assert os.path.exists(mask_path), "mask_path does not exist"
        assert os.path.exists(controlnet_path), "controlnet_path does not exist"
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        # get controlnet image width and height
        controlnet_img = open_controlnet_image(controlnet_path)
        controlnet_width, controlnet_height = controlnet_img.width, controlnet_img.height
        extra_args.update({
            "width" : controlnet_width,
            "height" : controlnet_height,
        })
        images_holder = []
        for prompt_1 in args_1:
            for prompt_2 in args_2:
                p1 = prompts[prompt_1]
                p2 = prompts[prompt_2]
                result = self.test_mask( 
                                    p1, p2,
                                    mask_path = mask_path,
                                    controlnet_path = controlnet_path,
                                    extra_args=extra_args,
                                    regional_prompter_args=regional_prompter_args,
                                    lora_stop_steps=lora_stop_steps,
                                    **test_dict
                                ).get_image()
                result.save(os.path.join(filepath, f"{prompt_1}_{prompt_2}_{suffix}.png"))
                if return_images:
                    images_holder.append(result)
        if return_images:
            return images_holder
        
    def test_and_save_division(self,
            args_1:List[str] = list(), args_2:List[str] = list(), suffix = "", filepath = './examples',
            controlnet_path:str = "",
            prompts:dict = dict(),
            return_images:bool = False,
            extra_args:dict = dict(),
            regional_prompter_args:Optional[dict] = None,
            lora_stop_steps:Tuple[int] = (10, 10),
            test_dict:Dict[str, Any] = dict(),
        ):
        """
        Tests division generation and save result to filepath
        """
        assert os.path.exists(controlnet_path), "controlnet_path does not exist"
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)
        controlnet_img = open_controlnet_image(controlnet_path)
        controlnet_width, controlnet_height = controlnet_img.width, controlnet_img.height
        extra_args.update({
            "width" : controlnet_width,
            "height" : controlnet_height,
        })
        images_holder = []
        for prompt_1 in args_1:
            for prompt_2 in args_2:
                p1 = prompts[prompt_1]
                p2 = prompts[prompt_2]
                result = self.test_division(
                                    p1, p2,
                                    controlnet_path = controlnet_path,
                                    extra_args=extra_args,
                                    regional_prompter_args=regional_prompter_args,
                                    lora_stop_steps=lora_stop_steps,
                                    **test_dict
                                ).get_image()
                result.save(os.path.join(filepath, f"{prompt_1}_{prompt_2}_{suffix}.png"))
                if return_images:
                    images_holder.append(result)
        if return_images:
            return images_holder
        

    def test_setup(
            self,
            args_1:List[str]=[],
            args_2:List[str]=[],
            suffix:str='example',
            filepath:str='./examples',
            mask_path:str='',
            controlnet_path:str='',
            prompts:Dict[str, str] = None, # mapping_to_test
            return_images:bool=False,
            regional_prompter_args:dict={},
            lora_stop_steps:Tuple[int]=(14, 7),
            extra_args:Dict[str, str]={},
            test_dict:Dict[str, Any]={},
        ) -> Union[List[Image.Image], None]:
        """
        Tests generation of images from prompts and saves them to filepath.
            @param instance: WebUIApi instance
            @param args_1: list of arguments for the first image, iterates over prompts
            @param args_2: list of arguments for the second image, iterates over prompts
            @param suffix: suffix for the saved image (args_1, args_2 does not have to be included)
            @param filepath: filepath to save the images
            @param mask_path: filepath to the mask image
            @param controlnet_path: filepath to the controlnet image
            @param prompts: dictionary of prompts to test, key is the name of the prompt, value is the prompt. args_1 and args_2 shoud contain the keys of the dictionary
            @param return_images: if True, returns the images as a list, else returns None
            @param regional_prompter_args: dictionary of arguments for regional prompter, key is the name of the argument, value is the value of the argument
                @example : {
                    "Division Mode" : "Horizontal", #can be 'Horizontal', 'Vertical', 'Random'
                    "Divide Ratio" : "1, 1", 
                    "Base Ratio" : "0.2", # Base prompt ratio
                    "LoRA in negative textencoder" : "0", # 0~1, default is 0
                    "LoRA in negative U-net" : "0", # 0~1, default is 0
                    "threshold" : "0", 
                    "Use Base Prompt" : False # First prompt will be used as base prompt if True,
                    "Use Common Prompt" : False,
                    "Use Common Negative Prompt" : False,
                }
            @param lora_stop_steps: tuple of steps to stop LoRA at, first element is first pass stop step, second element is second pass stop step (hires)
            @param extra_args: dictionary of extra arguments for generation, such as 'seed', 'hr_scale' etc. see WebUIApi.txt2img args for more info
                example : {
                    'seed' : 42
                }
            @param test_dict: dictionary of extra arguments for test functions, such as 'controlnet_preprocessor', 'controlnet_model' etc. see test functions for more info
                example : {
                    controlnet_preprocessor: str = "none",
                    controlnet_model: str = "control_v11p_sd15_openpose [cab7",
                    base_prompt: str = "masterpiece, night, line art, si",
                    common_prompt: str = "(best-quality:0.8), detailed fac",
                    negative_prompt: str = "(worst quality:0.8), verybadimage",
                }
            
            @return: list of images if return_images is True, else None
        """
        # check if all prompts are in args_1 and args_2
        assert prompts is not None and len(prompts) > 0, "prompts cannot be empty"
        assert len(args_1) > 0 and len(args_2) > 0, "args_1 and args_2 cannot be empty"
        assert all(prompt in prompts for prompt in args_1), "args_1 contains prompts not in prompts"
        assert all(prompt in prompts for prompt in args_2), "args_2 contains prompts not in prompts"
        
        # filepath
        if not os.path.exists(filepath):
            os.makedirs(filepath) # create directory if not exists
        
        # check if mask_path and controlnet_path exists
        
        result_container = []
        # mask path is invalid, skip mask
        if os.path.exists(mask_path):
            result = self.test_and_save(
                args_1=args_1,
                args_2=args_2,
                suffix=suffix + '_mask',
                filepath=filepath,
                mask_path=mask_path,
                controlnet_path=controlnet_path,
                prompts=prompts,
                return_images=return_images,
                regional_prompter_args=regional_prompter_args,
                lora_stop_steps=lora_stop_steps,
                extra_args=extra_args,
                test_dict=test_dict
            )
            if return_images:
                result_container.extend(result)
        else:
            print("mask_path does not exist, skipping mask")
        
        if os.path.exists(controlnet_path):
            modified_regional_prompter_args = regional_prompter_args.copy()
            modified_regional_prompter_args['Use Base Prompt'] = True
            result = self.test_and_save_division(
                args_1=args_1,
                args_2=args_2,
                suffix=suffix + '_division',
                filepath=filepath,
                controlnet_path=controlnet_path,
                prompts=prompts,
                return_images=return_images,
                regional_prompter_args=modified_regional_prompter_args,
                lora_stop_steps=lora_stop_steps,
                extra_args=extra_args,
                test_dict=test_dict
            )
            if return_images:
                result_container.extend(result)
        else:
            print("controlnet_path does not exist, skipping division")
        
        # if empty, raise error
        if return_images and len(result_container) == 0:
            raise ValueError("No images generated, check if mask_path and controlnet_path exists")
        return result_container

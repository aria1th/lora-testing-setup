from typing import List
from library.webuiapi import WebUIApi, QueuedTaskResult, ControlNetUnit, raw_b64_img
import os
import time
from PIL import Image
from threading import Thread

def open_controlnet_image(path:str):
    control_ref_img = Image.open(path)
    control_ref_img = control_ref_img.convert("RGB")
    return control_ref_img

def open_mask_image(path:str):
    mask_ref_img = Image.open(path)
    mask_ref_img = mask_ref_img.convert("RGB")
    return mask_ref_img

def wrap_upload_loras(apiInstance:WebUIApi, lora_dir:str):
    apiInstance.set_overwrite(True) # set overwrite to True
    lora_name = os.path.basename(lora_dir).split(".")[0]
    apiInstance.upload_lora(lora_dir, "test_by_api") # upload lora model from target_path to server's test_lora dir
    return lora_name


def process_test_latent_couple(apiInstance: WebUIApi, control_ref_path:str, mask_ref_path:str, 
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

def wait_for_result(result:QueuedTaskResult, result_list:list,
                    check_interval:int=1):
    """wait for result to finish"""
    # function for thread
    while not result.is_finished():
        time.sleep(check_interval)
    result_list.append(result)

def process_test_and_wait(apiInstance:WebUIApi, control_ref_path:str, mask_ref_path:str, result_list:list, should_wait_finish:bool=True,
                          test_args:dict={}, target_func=process_test_latent_couple):
    """process test and wait for result to finish"""
    result_control_latent_couple = target_func(apiInstance, control_ref_path, mask_ref_path, **test_args)
    thread = Thread(target=wait_for_result, args=(result_control_latent_couple, result_list))
    thread.start()
    if should_wait_finish:
        thread.join()
    return result_control_latent_couple


def concat_image_horizontally(img_list:list):
    """
    Concat images horizontally
    """
    max_width = max([img.width for img in img_list])
    max_height = max([img.height for img in img_list])
    width = max_width * len(img_list)
    height = max_height
    concat_image = Image.new("RGB", (width, height))
    offset_x = 0
    for img in img_list:
        target_ratio = max_width / img.width
        target_height = int(img.height * target_ratio)
        resized_img = img.resize((max_width, target_height))
        concat_image.paste(resized_img, (offset_x, 0))
        offset_x += max_width
        
    return concat_image


  
def test_mask(instance,
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
                    "1, 1", # Divide Ratio
                    "0.2", # Base Ratio
                    False, # Use base prompt
                    False, # Use common prompt
                    False, # Use common negative prompt
                    "Latent", # Attention, Latent
                    False, # disable convert 'AND' to 'BREAK'
                    "0", # LoRA in negative textencoder
                    "0", # LoRA in negative U-net
                    "0", # threshold
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

def test_and_save(args_1:List[str] = list(), args_2:List[str] = list(), suffix = "", filepath = './examples',
                mask_path:str = "",
                controlnet_path:str = "",
                prompts:dict = dict(),
                return_images:bool = False):
    """
    Tests mask generation and save result to filepath
    """
    assert os.path.exists(mask_path), "mask_path does not exist"
    assert os.path.exists(controlnet_path), "controlnet_path does not exist"
    if not os.path.exists(filepath):
        os.makedirs(filepath, exist_ok=True)
    images_holder = []
    for prompt_1 in args_1:
        for prompt_2 in args_2:
            p1 = prompts[prompt_1]
            p2 = prompts[prompt_2]
            result = test_mask(p1, p2,
                               mask_path = mask_path,
                               controlnet_path = controlnet_path,
                               ).get_image()
            result.save(os.path.join(filepath, f"{prompt_1}_{prompt_2}_{suffix}.png"))
            if return_images:
                images_holder.append(result)
    if return_images:
        return images_holder
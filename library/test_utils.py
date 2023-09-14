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

def wait_for_result(result:QueuedTaskResult, result_list:list):
    """wait for result to finish"""
    # function for thread
    while not result.is_finished():
        time.sleep(1)
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

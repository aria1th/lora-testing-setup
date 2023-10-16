"""
Class and functions for inference.
"""
from typing import List, Dict, Union
from time import sleep
import tempfile
import os
import requests
from library.webuiapi import WebUIApi, ControlNetUnit, QueuedTaskResult, raw_b64_img
from library.test_utils import open_controlnet_image

def recursive_convert_path_to_base64(obj:Union[List, Dict]) -> Union[List, Dict]:
    """
    Recursively converts value of dict or list to base64 string if value is path to image and ends with .png or .jpg
    """
    if isinstance(obj, list):
        return [recursive_convert_path_to_base64(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: recursive_convert_path_to_base64(value) for key, value in obj.items()}
    elif isinstance(obj, str):
        if os.path.exists(obj) and os.path.isfile(obj) and (obj.endswith('.png') or obj.endswith('.jpg')):
            return raw_b64_img(open_controlnet_image(obj))
        # check if image is link from internet
        try:
            if not obj.startswith('http'):
                return obj # not a link
            response = requests.get(obj, timeout=5)
            assert response.status_code == 200, f"Invalid url {obj}"
            # save to temp file and replace input_image with temp file path
            with tempfile.NamedTemporaryFile() as temp_file:
                temp_file.write(response.content)
                return raw_b64_img(open_controlnet_image(temp_file.name))
        except Exception as exception:
            raise ValueError(f"Invalid input image {obj}") from exception
    return obj

class InferenceSetup:
    """
    Class that loads inference setup from config file (contains prompts to test, webui address, etc.)
    """
    def __init__(self, webui_addr:str = "", webui_auth:str = "",
                 
                 ) -> None:
        self.instance = WebUIApi(webui_addr, webui_auth)
        if webui_auth and len(webui_auth.split(':')) == 2:
            self.instance.set_auth(webui_auth.split(':')[0], webui_auth.split(':')[1])
            
    def inference(self, settings:List[dict], should_wait:bool = True, sleep_interval:int=5) -> List[dict]:
        """
        Inference a list of settings (per image) then return the result.
        """
        result_container = []
        if should_wait:
            from tqdm import tqdm # in-context import for jupyter notebook #NoQA pylint: disable=import-outside-toplevel
            pbar = tqdm(total=len(settings))
        else:
            pbar = None
        for setting in settings:
            result_container.append(self.infernce_single_setting(setting))
            if should_wait:
                self.wait_until_finished(result_container[-1], sleep_interval)
                pbar.update(1)
        return result_container

    def wait_until_finished(self, result:QueuedTaskResult, sleep_interval:int=5) -> None:
        """
        Wait until QueuedTaskResult is finished.
        """
        while True:
            if result.is_finished():
                break
            sleep(sleep_interval)
        return
        
        
    def infernce_single_setting(self, setting:dict):
        """
        Inference a single setting (per image) then return the result.
        
        @param setting: Dict containing setting for inference.
            @Optional key: controlnet_args: List of Dict containing controlnet args
            @key *others: Dict containing other args for inference(txt2img_task)
                @ values can be path to image, which will be converted to base64 string.
                
        @return QueuedTaskResult
        
        @example
            self.inference_single_setting({
                'controlnet_args': [
                    {
                        'input_image': 'path/to/image.png',
                        'model': 'model_name',
                        'module': 'module_name',
                    }, ... # other controlnet args
                    ],
                'seed': 0,
                'width': 512,
                'height': 512,
                'prompt': '1girl, masterpiece',
                'negative_prompt': '1boy',
            })
                    
        """
        setting = setting.copy()
        self.validate_controlnet_args(setting.get('controlnet_args', []))
        controlnet_args = self.pop_controlnet_args(setting)
        controlnet_args_processed = self.process_controlnet_image_from_str(controlnet_args)
        controlnet_units = [self.construct_controlnet(args) for args in controlnet_args_processed]
        arguments_cleared = recursive_convert_path_to_base64(setting)
        result = self.instance.txt2img_task(
            controlnet_units=controlnet_units,
            **arguments_cleared
        )
        return result
   
    @staticmethod
    def construct_controlnet(controlnet_setting:dict) -> ControlNetUnit:
        """
        Construct a ControlNetUnit from controlnet_setting
        
        @param controlnet_setting: Dict containing controlnet args
            @key input_image: str, path to input image
            @key model: str, name of model
            @key module: str, name of module, can be 'none'
            @key kwargs: Dict, kwargs for ControlNetUnit
                @example
                    weight: float = 1,
                    resize_mode: str = "Resize and Fill",
                    lowvram: bool = False,
                    processor_res: int = 512,
                    threshold_a: float = 64,
                    threshold_b: float = 64,
                    guidance: float = 1,
                    guidance_start: float = 0,
                    guidance_end: float = 1,
                    control_mode: int = 0,
                    pixel_perfect: bool = False,
                    guessmode: int = None

        @return ControlNetUnit
        """
        # constructs and returns a ControlNetUnit
        return ControlNetUnit(
            input_image=controlnet_setting.get('input_image'), # str
            model=controlnet_setting.get('model'), # str
            module=controlnet_setting.get('module'), # str
            **controlnet_setting.get('kwargs', {})
        )

    def pop_controlnet_args(self, setting:dict) -> List[Dict]:
        """
        Pop controlnet args from setting dict
        Setting dict may contain 'controlnet_args' key.
        setting['controlnet_args'] should be List of Dict
            Dict should contain input_image, model, module args.
                input_image should be valid path locally, which will be converted to base64 string
        """
        controlnet_args = setting.pop('controlnet_args', [])
        try:
            self.validate_controlnet_args(controlnet_args)
        except (ValueError, FileNotFoundError) as exception:
            raise (f"Error when validating controlnet_args: {exception}") from exception
        return self.process_controlnet_image_from_str(controlnet_args) # convert images args to PIL Image

    def validate_controlnet_args(self, controlnet_args:List[Dict]) -> None:
        """
        Validate controlnet args.
            @key input_image: str, path to input image or url to image
            @key model: str, name of model
            @key module: str, name of module, can be 'none'
        Throws ValueError if invalid
        """
        for args in controlnet_args:
            if not args.get('input_image'):
                # check if file exists
                if not os.path.exists(args.get('input_image')):
                    # check if image is link from internet
                    try:
                        response = requests.get(args.get('input_image'), timeout=5)
                        assert response.status_code == 200, f"Invalid url {args.get('input_image')}"
                        # save to temp file and replace input_image with temp file path
                        with tempfile.NamedTemporaryFile() as temp_file:
                            temp_file.write(response.content)
                            args['input_image'] = temp_file.name
                    except Exception as exception:
                        raise ValueError(f"Invalid input image {args.get('input_image')}") from exception
                    raise FileNotFoundError(f"File {args.get('input_image')} not found")
            if not args.get('model'):
                raise ValueError("Model not specified")
            if not args.get('module_args'):
                raise ValueError("Module args not specified")
        return

    def process_controlnet_image_from_str(self, controlnet_args:List[Dict]) -> List[Dict]:
        """
        Process image to PIL Image from string(path). This should be called after validate_controlnet_args
        """
        for args in controlnet_args:
            if not args.get('input_image'):
                continue
            if not os.path.exists(args.get('input_image')):
                raise FileNotFoundError(f"File {args.get('input_image')} not found")
            args['input_image'] = open_controlnet_image(args['input_image'])
        return controlnet_args
    
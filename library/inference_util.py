"""
Class and functions for inference.
"""
import tempfile
import os
import re
import json
import yaml
import toml
import requests
from itertools import product
from typing import List, Dict, Union, Generator, Any
from os import PathLike
from time import sleep
from PIL import Image
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

class SettingLike(Dict[str, Union[Dict, str, int, float]]):
    """
    Class that represents a setting.
    """
    controlnet_args: List[Dict[str, Union[str, int, float]]]
    seed: int
    width: int
    height: int
    prompt: str
    negative_prompt: str
    
class InferenceInterface:
    """
    Interface for inference.
    Implements construct_controlnet, _load_filetype
    """
    def __init__(self) -> None:
        raise NotImplementedError #Should not be used
    
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
    @staticmethod
    def _load_filetype(config:Union[str, PathLike]) -> List[dict]:
        """
        Load config from file based on file extension.
        Handles .txt, .json,.jsonl, .yaml, .yml, .toml
        """
        if not os.path.exists(config):
            raise FileNotFoundError(f"File {config} not found")
        if config.endswith('.txt'):
            # load from txt file, each line is <prompt> --n <negative_prompt> --seed <seed> --width <width> --height <height>
            # --<args> <value> is optional
            settings = []
            with open(config, 'r', encoding='utf-8') as file:
                args_group = {
                    'prompt': None,
                    'negative_prompt': None,
                    'seed': None,
                    'width': None,
                    'height': None,
                } # match regex --<args> <value>
                for line in file.readlines():
                    # catch each args
                    for key in args_group.keys():
                        match = re.search(f"--{key} (.+?) ", line)
                        if match:
                            args_group[key] = match.group(1)
                    # if line is empty, skip
                    if not line.strip():
                        continue
                    # get normal prompt
                    prompt = re.search(r"^(.+?) --", line) # match regex <prompt> --
                    if prompt or args_group['prompt']:
                        args_group['prompt'] = prompt.group(1)
                        # prompt exists, add to settings
                        settings.append(args_group.copy())
                        continue
        elif config.endswith('.json'):
            # simple json file, load as json
            with open(config, 'r', encoding='utf-8') as file:
                settings = json.load(file)
        elif config.endswith('.jsonl'):
            # jsonl file, load as jsonl
            with open(config, 'r', encoding='utf-8') as file:
                settings = [json.loads(line) for line in file.readlines()]
        elif config.endswith('.yaml') or config.endswith('.yml'):
            # yaml file, load as yaml
            with open(config, 'r', encoding='utf-8') as file:
                settings = yaml.safe_load(file)
        elif config.endswith('.toml'):
            # toml file, load as toml
            with open(config, 'r', encoding='utf-8') as file:
                settings = toml.load(file)
        else:
            raise ValueError(f"Invalid config file {config}, should be .txt, .json, .jsonl, .yaml, .yml, .toml")
        return settings

class InferenceSetupFactory(InferenceInterface):
    """
    class that creates InferenceSetup from config file.
    There can be default config file, and modifiable config file, for combinational inference.
    
    example:
        # this handles default config (which is for InferenceSetup config)
        factory = InferenceSetupFactory(default_config='default_config.json', modifiable_config='modifiable_config.json')
        factory.inference() # this will generate all possible combinations from modifiable config, then combine with default config.
        # modifiable config handles all modifiable args, as raw string regex
        # for example, if '$1' is found in default config, and modifiable config contains {'$1': ['1girl', '1boy']}, then it will be replaced with '1girl' and '1boy'.
        # The factory tries to generate all possible combinations from modifiable config, then combine with default config.
        # if prompt is not found, then it will be skipped.

    """
    def __init__(self, default_config:Union[str, PathLike], modifiable_config:Union[str, PathLike]) -> None:
        """
        @param default_config: Path to default config file
        @param modifiable_config: Path to modifiable config file
        """
        if not os.path.exists(default_config):
            raise FileNotFoundError(f"Default config file {default_config} not found")
        if not os.path.exists(modifiable_config):
            raise FileNotFoundError(f"Modifiable config file {modifiable_config} not found")
        self.default_config_path = default_config
        self.modifiable_config_path = modifiable_config

    def generator(self) -> List['InferenceSetup']:
        """
        Yield all possible combinations of InferenceSetup
        """
        default_config = self.load_config(self.default_config_path)
        if not isinstance(default_config, list):
            # singletons, convert to list
            default_config = [default_config]
        modifiable_config = self.load_config(self.modifiable_config_path)
        assert isinstance(modifiable_config, dict), f"Modifiable config should be dict, got {type(modifiable_config)}"
        # keys will be 'prompt', 'negative_prompt', 'seed', 'width', 'height'... etc to work on
        # {'prompt' : {'$1': ['1girl', '1boy']}, 'negative_prompt': {'$1': ['1girl', '1boy']}, ...} if value is string (in default config)
        # {'seed' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'width': [512, 1024, 2048], ...} if value is list (in default config)
        # if recursive dict, then use yield from
        # {'controlnet_args' : {'seed' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'width': [512, 1024, 2048], ...}}
        raise NotImplementedError # TODO: implement this
        
    def _generator_part(self, target_dict:dict, modifying_dict:dict) -> List[dict]:
        """
        Yield all possible combinations of target_dict with modifying_dict
        """
        raise NotImplementedError # TODO: implement this 

    def load_config(self, config:Union[str, PathLike]) -> Union[List[Dict], Dict[str, str]]:
        """
        Loads config from file.
        """
        config_dict = {}
        if isinstance(config, str):
            config_dict = self._load_filetype(config)
        elif isinstance(config, PathLike):
            config_dict = self._load_filetype(str(config))
        else:
            raise ValueError(f"Invalid config {config}")
        return config_dict

class SimpleInferenceWithReplace:
    """
    Class to handle InferenceSetup object to replace values.
    """
    def __init__(self, setting_json:Dict[Union[re.Pattern, str], List[str]]) -> None:
        """
        @param setting_json: Dict containing key:[values] to replace values
            @key: string to replace(keys)
            @value: List of values to replace
        """
        if isinstance(setting_json, str):
            if os.path.exists(setting_json):
                with open(setting_json, 'r', encoding='utf-8') as file:
                    setting_json = json.load(file)
            else:
                setting_json = json.loads(setting_json)
        self.setting_json = setting_json
    
    def generator(self, inference_setup:SettingLike) -> Generator[SettingLike, None, None]:
        """
        Yield all possible combinations of InferenceSetup
        """
        # Behavior will be different depends on 'recursive' key - if dict is in setting_json, flatten then search if matching keys to replace exists
        # if not, then we can just use existing matches and use product to generate all possible combinations
        if any(isinstance(v, dict) for v in self.setting_json):
            # check if key is in inference_setup
            recursive_keys = [k for k, v in self.setting_json.items() if isinstance(v, dict)]
            for key in recursive_keys:
                if key not in inference_setup:
                    # key not found, skip
                    continue
            raise NotImplementedError("Recursive dict is not implemented yet") # TODO: implement this
        else:
            # simply use product to generate all possible combinations
            # convert {k:[v]} to {k:[(k,v)]} for product
            setting_json = {k:[(k, v) for v in v_list] for k, v_list in self.setting_json.items()}
            # generate all possible combinations with values
            for combination in product(*setting_json.values()):
                # replace values in inference_setup
                new_inference_setup = inference_setup.copy()
                for key, value in combination:
                    if key not in new_inference_setup:
                        continue
                    new_inference_setup[key] = value
                yield new_inference_setup
    
    def generator_multiple(self, inference_setups:List[SettingLike]) -> Generator[List[SettingLike], None, None]:
        """
        Yield all possible combinations of InferenceSetup.
        [a, b, c, d] -> [Generator(a), Generator(b),...] would return list of new SettingLike or None object.
        For each generator, it may yield different amount of SettingLike. This will result [SettingLike, None, SettingLike...] for example.
        """
        generators = [self.generator(inference_setup) for inference_setup in inference_setups]
        # sequential yield, generate until all generators are exhausted
        while True:
            result_container = []
            for generator in generators:
                try:
                    result_container.append(next(generator))
                except StopIteration:
                    result_container.append(None)
            yield result_container
            if all(result is None for result in result_container):
                break
            
    def inference_setting(self, inference_setup:SettingLike, 
                  webui_addr:str, webui_auth:str = "") -> List[QueuedTaskResult]:
        """
        Inference a single InferenceSetup object.
        """
        inference_handler = InferenceSetup(webui_addr, webui_auth)
        results = []
        for new_inference_setup in self.generator(inference_setup):
            if not new_inference_setup:
                continue
            results.append(inference_handler.infernce_single_setting(new_inference_setup))
        return results
    
    def inference(self, inference:List[SettingLike],
                  webui_addr:str, webui_auth:str = "", debug:bool = False) -> List[List[QueuedTaskResult]]:
        """
        Inference a list of InferenceSetup object.
        """
        inference_handler = InferenceSetup(webui_addr, webui_auth)
        results:List[List[Any]] = []
        for new_inference_setups in self.generator_multiple(inference):
            if not new_inference_setups:
                continue
            partial_results = []
            for new_inference_setup in new_inference_setups:
                if not new_inference_setup:
                    partial_results.append(None)
                else:
                    if debug:
                        partial_results.append(new_inference_setup)
                    else:
                        partial_results.append(inference_handler.infernce_single_setting(new_inference_setup))
            if any(result is not None for result in partial_results):
                results.append(partial_results)
        return results

class InferenceSetup(InferenceInterface):
    """
    Class that loads inference setup from config file (contains prompts to test, webui address, etc.)
    """
    def __init__(self, webui_addr:str = "", webui_auth:str = "",
                 
                 ) -> None:
        self.instance = WebUIApi(baseurl=webui_addr)
        if webui_auth and len(webui_auth.split(':')) == 2:
            self.instance.set_auth(webui_auth.split(':')[0], webui_auth.split(':')[1])
        elif webui_auth: # invalid format, raise error
            raise ValueError(f"Invalid webui_auth format. Should be username:password, got {webui_auth}")

    def inference(self, settings:List[SettingLike], should_wait:bool = True, sleep_interval:int=5) -> List[dict]:
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
  
    def from_config(self, config:Union[str, PathLike], should_wait:bool = True, sleep_interval:int=5) -> List[dict]:
        """
        Load config from file then inference.
        """
        settings = self.load_config(config)
        return self.inference(settings, should_wait, sleep_interval)

    def load_config(self, config:Union[str, PathLike]) -> List[dict]:
        """
        Load config from file.
        """
        config_dict = {}
        if isinstance(config, str):
            config_dict = self._load_filetype(config)
        elif isinstance(config, PathLike):
            config_dict = self._load_filetype(str(config))
        else:
            raise ValueError(f"Invalid config {config}")
        # if webui_addr or webui_auth is specified in config, override the one in constructor
        if config_dict.get('webui_addr') or config_dict.get('webui_auth'):
            new_instance = WebUIApi(baseurl=config_dict.get('webui_addr', ''))
            if config_dict.get('webui_auth', '') and len(config_dict.get('webui_auth').split(':')) == 2:
                new_instance.set_auth(config_dict.get('webui_auth').split(':')[0], config_dict.get('webui_auth').split(':')[1])
            self.instance = new_instance
        return config_dict.get('settings', [])

    def wait_until_finished(self, result:QueuedTaskResult, sleep_interval:int=5) -> None:
        """
        Wait until QueuedTaskResult is finished.
        """
        while True:
            if result.is_finished():
                break
            sleep(sleep_interval)
        return

    def infernce_single_setting(self, setting:SettingLike):
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

    def pop_controlnet_args(self, setting:SettingLike) -> List[Dict]:
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
            if not args.get('module'):
                raise ValueError("Module not specified")
        return

    def process_controlnet_image_from_str(self, controlnet_args:List[Dict]) -> List[Dict]:
        """
        Process image to PIL Image from string(path). This should be called after validate_controlnet_args
        """
        for args in controlnet_args:
            if not args.get('input_image'):
                continue
            if isinstance(args.get('input_image'), str) and os.path.exists(args.get('input_image')):
                args['input_image'] = open_controlnet_image(args['input_image'])
            elif isinstance(args.get('input_image'), Image.Image):
                pass
            else: #maybe no file exists
                if not os.path.exists(args.get('input_image')):
                    raise FileNotFoundError(f"File {args.get('input_image')} not found")
                raise ValueError(f"Invalid input image {args.get('input_image')}") # should not happen
        return controlnet_args

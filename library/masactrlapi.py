from .webuiapi import WebUIApi, HiResUpscaler, ControlNetUnit, QueuedTaskResult
from .ipadapterapi import IPAdapterAPI
from typing import List
from requests import post
from json import loads
from random import Random
import time

RandomGenerator = Random(42) # Deterministic random generator

class Status:
    LOGGING = 0 # logging
    RECON = 1 # reconstruction
    LOGRECON = 2 # logging and recon
    IDLE = 3 # idle
    # categories
    LOG = {LOGGING, LOGRECON}

class MasaCtrlApi(WebUIApi):
    """
    Mutual-Self-Attention based Control API
    
    """
    MASACTRL_MODES = {
        Status.IDLE: 3,
        Status.LOGGING: 0,
        Status.RECON:1,
        Status.LOGRECON: 2
    }
    def __init__(self, *args, **kwargs):
        self.seed = -1 # log seed
        self.masactrl_state = Status.IDLE
        self.last_generation_w_h = (0, 0) # last generation width and height
        super().__init__(*args, **kwargs)
        

    def get_tokens(self, prompt:str) -> int:
        """
        Calculates the number of tokens in the prompt.
        curl -X POST "http://localhost:7861/sdapi/v1/count_tokens" -H  "accept: application/json" -H  "Content-Type: application/x-www-form-urlencoded" -d "prompt=Hello%20World"
        """
        response = self.session.post(
            self.real_url + "/sdapi/v1/count_tokens",
            data={"prompt": prompt}
        )
        # check if the request was successful
        if response.status_code != 200:
            raise ConnectionError(f"Request failed with status code {response.status_code}, server requires sd-webui-uploader(https://github.com/aria1th/webui-model-uploader) installed." + response.text)
        # parse the response
        return loads(response.text)["token_count"]
    
    def reset(self):
        """
        Reset the seed and masactrl_state.
        """
        self.seed = -1
        self.masactrl_state = Status.IDLE
        self.last_generation_w_h = (0, 0)
        
    def pop_args(self, kwargs:dict):
        """
        Remove the masa ctrl args from kwargs.
        """
        # pop masa ctrl args
        args_to_pop = {
            'masactrl_mode',
            'fixed_prompt',
            'append_prompt',
        }
        for arg in args_to_pop:
            if arg in kwargs:
                kwargs.pop(arg)
        return kwargs # return the modified kwargs
    
    def txt2img(self, *args, **kwargs):
        return self._common_txt2img(*args, **kwargs, is_task=False)
        
    def txt2img_task(self, *args, **kwargs):
        return self._common_txt2img(*args, **kwargs, is_task=True)
    
    def generate_images(self, base_prompt:str = "", append_prompt_list:List[str] = [], use_task:bool=True,
                        enable=True, # enable the masactrl, this is for debugging
                        *args, **kwargs):
        """
        Generate images from text using the Mutual-Self-Attention based Control API.
        Requires sd-webui-uploader in server side.
            @param base_prompt: The base prompt, this will be the first prompt.
            @param append_prompt_list: The list of prompts to append to the base prompt.
            @param use_task: If True, it will use the task version of txt2img, else it will use the normal version. Requires Agent Scheduler in server side.
        """
        if use_task:
            func_to_call = self.txt2img_task
        else:
            func_to_call = self.txt2img
        length = len(append_prompt_list) # get the length of the append_prompt_list
        # first one is LOG, next is LOGRECON as continued
        modes = [Status.LOGGING] + [Status.LOGRECON] * (length - 1)
        # generate the images
        images = []
        # if task, it returns QueuedTaskResult, else it returns the image
        for i in range(length):
            kwargs['masactrl_mode'] = modes[i] if enable else Status.IDLE
            kwargs['fixed_prompt'] = base_prompt
            kwargs['append_prompt'] = append_prompt_list[i]
            if use_task:
                queued_task:QueuedTaskResult = func_to_call(*args, **kwargs)
                images.append(queued_task)
            else:
                image = func_to_call(*args, **kwargs).images[0] # get the first image
                images.append(image)
        if not use_task:
            return images # does not have to wait
        # wait for the tasks to finish
        images:List[QueuedTaskResult] = images # type hint
        while True:
            finished = True
            for task in images:
                if not task.is_finished():
                    finished = False
                    break
            if finished:
                break
            time.sleep(1) # wait for 1 second
        return [task.get_image() for task in images]
    
    def _common_txt2img(self,
            enable_hr=False,
            denoising_strength=0.7,
            firstphase_width=0,
            firstphase_height=0,
            hr_scale=2,
            hr_upscaler=HiResUpscaler.Latent,
            hr_second_pass_steps=0,
            hr_resize_x=0,
            hr_resize_y=0,
            prompt="",
            styles=[],
            seed=-1,
            subseed=-1,
            subseed_strength=0.0,
            seed_resize_from_h=0,
            seed_resize_from_w=0,
            sampler_name=None,  # use this instead of sampler_index
            batch_size=1,
            n_iter=1,
            steps=None,
            cfg_scale=7.0,
            width=512,
            height=512,
            restore_faces=False,
            tiling=False,
            do_not_save_samples=False,
            do_not_save_grid=False,
            negative_prompt="",
            eta=1.0,
            s_churn=0,
            s_tmax=0,
            s_tmin=0,
            s_noise=1,
            override_settings={},
            override_settings_restore_afterwards=True,
            script_args=None,  # List of arguments for the script "script_name"
            script_name=None,
            send_images=True,
            save_images=False,
            alwayson_scripts={},
            controlnet_units: List[ControlNetUnit] = [],
            sampler_index=None,  # deprecated: use sampler_name
            use_deprecated_controlnet=False,
            use_async=False,
            checkpoint_name:str="",
            masactrl_mode=Status.IDLE,
            fixed_prompt='',
            append_prompt='',
            is_task=False,
        ):
        """
        Generate images from text using the Mutual-Self-Attention based Control API.
        Requires sd-webui-uploader in server side.
        
        Arguments to add :
            @param masactrl_mode: The mode of the masactrl, see Status for more details.
            @param fixed_prompt: The fixed prompt, this will be the first prompt.
            @param append_prompt: The prompt to append to the fixed prompt.
            
        Don't ask why I didn't use **kwargs, it is better for debugging...
        """
        kwargs = locals() # get all the arguments as a dictionary, this should be first line in the function
        args_to_get = {
            'enable_hr', 'denoising_strength', 'firstphase_width', 'firstphase_height', 'hr_scale', 
            'hr_upscaler', 'hr_second_pass_steps', 'hr_resize_x', 'hr_resize_y', 'prompt', 'styles', 
            'seed', 'subseed', 'subseed_strength', 'seed_resize_from_h', 'seed_resize_from_w', 'sampler_name', 
            'batch_size', 'n_iter', 'steps', 'cfg_scale', 'width', 'height', 'restore_faces', 'tiling', 'do_not_save_samples', 
            'do_not_save_grid', 'negative_prompt', 'eta', 's_churn', 's_tmax', 's_tmin', 's_noise', 'override_settings', 
            'override_settings_restore_afterwards', 'script_args', 'script_name', 'send_images', 'save_images', 'alwayson_scripts', 
            'controlnet_units', 'sampler_index', 'use_deprecated_controlnet', 'use_async', 'checkpoint_name', 'masactrl_mode', 
            'fixed_prompt', 'append_prompt', 'is_task'
        } # thank you copilot
        kwargs = {key: kwargs[key] for key in args_to_get} # get the arguments we need
        assert masactrl_mode in {Status.IDLE, Status.LOGGING, Status.RECON, Status.LOGRECON}, "masactrl_mode must be one of the Status"
        #print(kwargs)
        if is_task:
            func_to_call = super().txt2img_task
        else:
            func_to_call = super().txt2img
            kwargs.pop('checkpoint_name') # remove checkpoint_name from kwargs, since it is not supported in txt2img
        kwargs.pop('is_task') # remove is_task from kwargs
        masactrl_mode = kwargs.get('masactrl_mode', Status.IDLE)
        if masactrl_mode == Status.IDLE: # use the default txt2img
            self.reset()
            kwargs = self.pop_args(kwargs)
            return func_to_call(**kwargs)
        else:
            # ========== Width and Height Assertion ==========
            width = kwargs.get('width', 0)
            height = kwargs.get('height', 0)
            assert width > 0 and height > 0, "Width and Height must be greater than 0" # common assertion
            if masactrl_mode not in Status.LOG and self.last_generation_w_h != (0,0) and self.last_generation_w_h != (width, height):
                # for reconstruction, the width and height must be the same as the last generation
                raise Exception(f"Width and Height must be the same as the last generation: {self.last_generation_w_h}")
            # ========== Seed ==========
            seed = kwargs.get('seed', -1)
            if seed == -1:
                if self.seed != -1:
                    seed = self.seed # use the last seed
                else:
                    seed = RandomGenerator.randint(2, 2**32-1) # generate a new seed
                    self.seed = seed # save the seed
        # kwargs requires 'fixed_prompt' and 'append_prompt'
        fixed_prompt = kwargs.get('fixed_prompt', '')
        append_prompt = kwargs.get('append_prompt', '')
        # if both are empty, use the default txt2img
        if fixed_prompt == '' and append_prompt == '': # use the default txt2img
            print("Masactrl mode was given but no prompt(fixed_prompt or append_prompt) was given, use the default txt2img")
            self.reset()
            kwargs = self.pop_args(kwargs)
            return func_to_call( **kwargs)
        # fixed prompt can be just 'prompt' which is index 9
        if kwargs.get('prompt', '') != '':
            fixed_prompt = kwargs.get('prompt', '')
        # ========== Prompt ==========
        token_count = self.get_tokens(fixed_prompt) # get the number of tokens in the prompt
        prompt = ','.join([fixed_prompt, append_prompt])
        # ========== override kwargs ==========
        kwargs_to_override = {
            'seed': seed,
            'prompt': prompt,
        }
        self.seed = seed # save the seed
        # ========== Construct masactrl alwayson scripts ==========
        alwayson_scripts_masactrl = {
            'masa control': {
                "args" : [
                    MasaCtrlApi.MASACTRL_MODES[masactrl_mode],
                    5.0, # steps to control from
                    10.0, # layers to control from
                    0.1, # threshold
                    str(token_count) # tokens to fix, this has to be a string
                ]
            }
        }
        # ========== Construct alwayson scripts ==========
        alwayson_scripts = kwargs.get('alwayson_scripts', {})
        alwayson_scripts.update(alwayson_scripts_masactrl)
        kwargs_to_override['alwayson_scripts'] = alwayson_scripts
        # ========== Update kwargs ==========
        kwargs.update(kwargs_to_override)
        # ========== Call txt2img ==========
        self.last_generation_w_h = (width, height)
        kwargs = self.pop_args(kwargs)
        return func_to_call(**kwargs)
    
def wait_for_task(task:QueuedTaskResult):
    """
    Wait for the task to finish.
    """
    while not task.is_finished():
        time.sleep(1)
    return task.get_image()

class MasaCtrlAndIP(MasaCtrlApi):
    """
    Mutual-Self-Attention based Control API with IP Adapter API.
    """
    
    def __init__(self, *args, **kwargs):
        self.ip_adapter_api = None
        super().__init__(*args, **kwargs)
        
    def attach_ip(self, ip_adapter_api:IPAdapterAPI):
        """
        Attach the IP Adapter API.
        """
        self.ip_adapter_api = ip_adapter_api
        
    def _common_txt2img(self, *args, **kwargs):
        kwargs = self.ip_adapter_api._hijack_args(**kwargs)
        return super()._common_txt2img(*args, **kwargs)
    
    
    def generate_images(self, base_prompt:str = "", append_prompt_list:List[str] = [], use_task:bool=False,
                        enable=True, # enable the masactrl, this is for debugging
                        *args, **kwargs):
        """
        Generate images from text using the Mutual-Self-Attention based Control API.
        Requires sd-webui-uploader in server side.
            @param base_prompt: The base prompt, this will be the first prompt.
            @param append_prompt_list: The list of prompts to append to the base prompt.
            @param use_task: If True, it will use the task version of txt2img, else it will use the normal version. Requires Agent Scheduler in server side.
        """
        if use_task:
            func_to_call = self.txt2img_task
        else:
            func_to_call = self.txt2img
        # generate first image
        copied_kwargs = kwargs.copy()
        copied_kwargs['fixed_prompt'] = base_prompt
        copied_kwargs['append_prompt'] = ""
        copied_kwargs['masactrl_mode'] = Status.IDLE
        image_base = func_to_call(*args, **copied_kwargs).images[0]
        if use_task:
            image_base = wait_for_task(image_base)
        self.ip_adapter_api.previous_image = image_base # set previous image
        length = len(append_prompt_list) # get the length of the append_prompt_list
        # first one is LOG, next is LOGRECON as continued
        modes = [Status.LOGGING] + [Status.LOGRECON] * (length - 1)
        # generate the images
        images = []
        # if task, it returns QueuedTaskResult, else it returns the image
        for i in range(length):
            kwargs['masactrl_mode'] = modes[i] if enable else Status.IDLE
            kwargs['fixed_prompt'] = base_prompt
            kwargs['append_prompt'] = append_prompt_list[i]
            if use_task:
                queued_task:QueuedTaskResult = func_to_call(*args, **kwargs)
                # wait for the task to finish because ip-adapter requires the previous image to be passed to the next image
                image = wait_for_task(queued_task)
                images.append(image)
                self.ip_adapter_api.previous_image = image # set previous image
            else:
                image = func_to_call(*args, **kwargs).images[0] # get the first image
                images.append(image)
                self.ip_adapter_api.previous_image = image # set previous image
        return images
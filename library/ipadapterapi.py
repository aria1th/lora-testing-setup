"""
Handler for IP Adapter API
The webui instance requires IP-Adpater model and preprocessor.
"""
from .webuiapi import ControlNetUnit, HiResUpscaler, WebUIApi, QueuedTaskResult, WebUIApiResult
from PIL import Image
from typing import List, Optional, Dict, Any, Union

class IPAdapterAPI(WebUIApi):
    """
    Handler for IP Adapter API
    """
    previous_image: Optional[Image.Image] = None
    ip_controlnet_args: Dict[str, Any] = {}
    checked_compat: bool = False
    default_ip_controlnet_args: Dict[str, Union[str, bool, int, float]] = {
        'module' :"ip-adapter_clip_sd15",
        'model' : "ip-adapter_sd15 [6a3f6166]",
        'weight':  1,
        'resize_mode': "Resize and Fill",
        'lowvram': False,
        'processor_res': 512,
        'threshold_a': 64,
        'threshold_b': 64,
        'guidance': 1,
        'guidance_start': 0,
        'guidance_end': 1,
        'control_mode': 0,
        'pixel_perfect': False,
        'guessmode': None
    }
    default_reference_controlnet_kwargs: Dict[str, Any] = {
        'model' :"None",
        'module' : "reference_only",
        'weight':  1,
        'resize_mode': "Resize and Fill",
        'lowvram': False,
        'processor_res': 512,
        'threshold_a': 0.5,
        'threshold_b': 64,
        'guidance': 0.6,
        'guidance_start': 0,
        'guidance_end': 1,
        'control_mode': 0,
        'pixel_perfect': False,
        'guessmode': None
    }

    def __init__(self, host="127.0.0.1", port=7860, baseurl=None, sampler="Euler a", steps=20, use_https=False, username=None, password=None,
                 use_reference_controlnet:bool=False,
                 reference_controlnet_kwargs:Dict[str, Any]={},
                 **controlnet_kwargs):
        """
        Initialize IP Adapter API handler
        @param (see WebUIApi.__init__)
        @param controlnet_kwargs: Keyword arguments to pass to ControlNetUnit
            @keyword module: Module to use
            @keyword model: Model to use
            @keyword weight: Weight
            @keyword resize_mode: Resize mode
            @keyword lowvram: Low VRAM mode
            @keyword processor_res: Processor resolution
            @keyword threshold_a: Threshold A
            @keyword threshold_b: Threshold B
            @keyword guidance: Guidance
            @keyword guidance_start: Guidance start
            @keyword guidance_end: Guidance end
            @keyword control_mode: Control mode
            @keyword pixel_perfect: Pixel perfect
            @keyword guessmode: Guess mode
        """
        super().__init__(host, port, baseurl, sampler, steps, use_https, username, password)
        self.ip_controlnet_args = self.default_ip_controlnet_args.copy()
        for k in controlnet_kwargs:
            if k not in self.ip_controlnet_args:
                raise ValueError(f"Unknown argument {k}, allowed arguments are {list(self.ip_controlnet_args.keys())}")
        self.ip_controlnet_args.update(controlnet_kwargs)
        self.use_reference_controlnet = use_reference_controlnet
        self.reference_controlnet_kwargs = self.default_reference_controlnet_kwargs.copy()
        self.reference_controlnet_kwargs.update(reference_controlnet_kwargs)
        

    def _warn_incompatible(self) -> None:
        """
        Warn for incompatible extensions in UI side
            negpip is not compatible with IP Adapter
        """
        if self.checked_compat:
            return
        scripts = self.get_scripts()
        scripts_list = scripts['txt2img']
        if 'controlnet' not in scripts_list:
            raise ValueError("IP Adapter API requires controlnet extension")
        if 'negpip' in scripts_list:
            # https://github.com/hako-mikan/sd-webui-negpip/issues/13
            print("IP Adapter API is not compatible with negpip extension")
        self.checked_compat = True

    def _hijack_args(self, **kwargs) -> Dict[str, Any]:
        """
        Appends previous image to kwargs
        """
        if self.previous_image is None:
            # No previous image, just return kwargs
            return kwargs
        controlnet_module_lists = kwargs.get('controlnet_units', [])
        if controlnet_module_lists is None:
            controlnet_module_lists = [] #initialize
        else:
            controlnet_module_lists = controlnet_module_lists.copy() # copy to prevent in-place modification
        controlnet_unit = ControlNetUnit(**self.ip_controlnet_args, input_image=self.previous_image)
        controlnet_module_lists.append(controlnet_unit)
        if self.use_reference_controlnet:
            reference_controlnet_unit = ControlNetUnit(**self.reference_controlnet_kwargs, input_image=self.previous_image)
            controlnet_module_lists.append(reference_controlnet_unit)
        kwargs['controlnet_units'] = controlnet_module_lists # in-place append should have worked before, but for safety
        return kwargs

    def txt2img(self, *args, **kwargs) -> Image:
        """
        Convert text to image
        If self.previous_image is not None, it will be used as the base image. (appended to controlnet image)
        """
        self._warn_incompatible()
        kwargs = self._hijack_args(**kwargs)
        return super().txt2img(*args, **kwargs)
    
    def txt2img_task(self, *args, **kwargs) -> str:
        """
        Convert text to image (async)
        If self.previous_image is not None, it will be used as the base image. (appended to controlnet image)
        """
        self._warn_incompatible()
        kwargs = self._hijack_args(**kwargs)
        return super().txt2img_task(*args, **kwargs)
    
    def generate_images(self, base_prompt:str = "", append_prompt_list:List[str] = [], use_task:bool=True,
                        *args, **kwargs) -> List[Image.Image]:
        """
        Generate images from prompt list
        
            @param base_prompt: Base prompt to use
            @param append_prompt_list: List of prompts to append to base_prompt
            @param use_task: Use async task or not
            @param args: Arguments to pass to txt2img or txt2img_task
            @param kwargs: Keyword arguments to pass to txt2img or txt2img_task
                @keyword base_image: Base image to use (overrides previous image)(optional)
                @keyword update_previous_image: Update previous image or not (default: True)
            
            @return: List of images
        """
        from tqdm import tqdm # noqa # pylint: disable=import-outside-toplevel
        import time # noqa # pylint: disable=import-outside-toplevel
        if 'base_image' in kwargs:
            self.previous_image = kwargs['base_image']
            kwargs.pop('base_image')
        update_previous_image = kwargs.pop('update_previous_image', 'base_image' not in kwargs)
        if use_task:
            func_to_use = self.txt2img_task
        else:
            func_to_use = self.txt2img
        images = []
        length = len(append_prompt_list)
        pbar = tqdm(total=length if self.previous_image is not None else length+1)
        if self.previous_image is None:
            # create image for the first time
            pbar.set_description("Generating base image")
            kwargs_copied = kwargs.copy()
            kwargs_copied['prompt'] = base_prompt
            result:Union[QueuedTaskResult, WebUIApiResult] = func_to_use(*args, **kwargs_copied)
            if use_task:
                while not result.is_finished():
                    time.sleep(1)
            result = result.get_image()
            pbar.update()
            self.previous_image = result # set previous image
        for i, _prompt in enumerate(append_prompt_list):
            prompt = ', '.join([base_prompt, _prompt])
            if length > 1:
                pbar.set_description(f"Generating image {i+1}/{length}")
            kwargs_copied = kwargs.copy()
            kwargs_copied['prompt'] = prompt
            kwargs_copied['controlnet_units'] = kwargs_copied.get('controlnet_units', []).copy() # copy to prevent in-place modification
            result:Union[QueuedTaskResult, WebUIApiResult] = func_to_use(*args, **kwargs_copied)
            pbar.update()
            # ip-adapter requires previous image to be passed to the next image
            if use_task:
                while not result.is_finished():
                    time.sleep(1)
            result = result.get_image()
            images.append(result)
            if update_previous_image:
                self.previous_image = result
        pbar.close()
        return images
    
    def animate(self, base_prompt:str, prompt_1:str, prompt_2:str, steps:int=10, use_task:bool=True, filename:str="animate.gif",
                *args, **kwargs) -> List[Image.Image]:
        """
        Generate images from prompt list
        
            @param base_prompt: Base prompt to use
            @param prompt_1: Prompt 1
            @param prompt_2: Prompt 2
            @param steps: Number of steps to animate
            @param use_task: Use async task or not
            @param args: Arguments to pass to txt2img or txt2img_task
            @param kwargs: Keyword arguments to pass to txt2img or txt2img_task
            
            @return: List of images
        """
        import random # noqa # pylint: disable=import-outside-toplevel
        seed = kwargs.get('seed', random.randint(1, 100000000))
        kwargs['seed'] = seed
        prompt_lists = []
        for i in range(steps):
            ratio, residual_ratio = i/steps, (steps-i)/steps
            prompt_step = f"({prompt_1} : {residual_ratio}), ({prompt_2} : {ratio})"
            prompt_lists.append(prompt_step)
        images:List[Image.Image] = self.generate_images(base_prompt, prompt_lists, use_task, *args, **kwargs)
        # animate to gif
        image_base = images[0]
        image_base.save(filename, save_all=True, append_images=images[1:], duration=100, loop=0)
        return images
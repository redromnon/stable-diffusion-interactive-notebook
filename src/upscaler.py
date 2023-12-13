from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import torch
import PIL
import numpy as np
import contextlib
from io import StringIO
from tqdm.auto import tqdm
import signal
import requests
import urllib.request
import urllib.parse
import os
import re

def download_file(
    link: str,
    path: str,
    block_size: int = 1024,
    force_download: bool = False,
    progress: bool = True,
    interrupt_check: bool = True,
) -> str:
    def truncate_string(string: str, length: int):
        length -= 5 if length - 5 > 0 else 0
        curr_len = len(string)
        new_len = len(string[: length // 2] + "(...)" + string[-length // 2 :])
        if new_len > curr_len:
            return string
        else:
            return string[: length // 2] + "(...)" + string[-length // 2 :]

    def remove_char(string: str, chars: list):
        for char in chars:
            string = string.replace(char, "")
        return string

    # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
    def google_drive_parse_url(url: str):
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        is_gdrive = parsed.hostname in ["drive.google.com", "docs.google.com"]
        is_download_link = parsed.path.endswith("/uc")

        if not is_gdrive:
            return is_gdrive, is_download_link

        file_id = None
        if "id" in query:
            file_ids = query["id"]
            if len(file_ids) == 1:
                file_id = file_ids[0]
        else:
            patterns = [r"^/file/d/(.*?)/view$", r"^/presentation/d/(.*?)/edit$"]
            for pattern in patterns:
                match = re.match(pattern, parsed.path)
                if match:
                    file_id = match.groups()[0]
                    break

        return file_id, is_download_link

    # source: https://github.com/wkentaro/gdown/blob/main/gdown/download.py
    def get_url_from_gdrive_confirmation(contents: str):
        url = ""
        for line in contents.splitlines():
            m = re.search(r'href="(/uc\?export=download[^"]+)', line)
            if m:
                url = "https://docs.google.com" + m.groups()[0]
                url = url.replace("&amp;", "&")
                break
            m = re.search('id="download-form" action="(.+?)"', line)
            if m:
                url = m.groups()[0]
                url = url.replace("&amp;", "&")
                break
            m = re.search('"downloadUrl":"([^"]+)', line)
            if m:
                url = m.groups()[0]
                url = url.replace("\\u003d", "=")
                url = url.replace("\\u0026", "&")
                break
            m = re.search('<p class="uc-error-subcaption">(.*)</p>', line)
            if m:
                error = m.groups()[0]
                raise RuntimeError(error)
        if not url:
            raise RuntimeError("Cannot retrieve the link of the file. ")
        return url

    def interrupt(*args):
        if os.path.isfile(filepath):
            os.remove(filepath)
        raise KeyboardInterrupt

    # create folder if not exists
    if not os.path.exists(path):
        os.makedirs(path)

    # check if link is google drive link
    if not google_drive_parse_url(link)[0]:
        response = requests.get(link, stream=True, allow_redirects=True)
    else:
        if not google_drive_parse_url(link)[1]:
            # convert to direct link
            file_id = google_drive_parse_url(link)[0]
            link = f"https://drive.google.com/uc?id={file_id}"
        # test if redirect is needed
        response = requests.get(link, stream=True, allow_redirects=True)
        if response.headers.get("Content-Disposition") is None:
            page = urllib.request.urlopen(link)
            link = get_url_from_gdrive_confirmation(str(page.read()))
            response = requests.get(link, stream=True, allow_redirects=True)

    if response.status_code == 404:
        raise FileNotFoundError(f"File not found at {link}")

    # get filename
    content_disposition = response.headers.get("Content-Disposition")
    if content_disposition:
        filename = re.findall(r"filename=(.*?)(?:[;\n]|$)", content_disposition)[0]
    else:
        filename = os.path.basename(link)

    filename = remove_char(
        filename, ["/", "\\", ":", "*", "?", '"', "'", "<", ">", "|", ";"]
    )
    filename = filename.replace(" ", "_")

    filepath = os.path.join(path, filename)

    # download file
    if os.path.isfile(filepath) and not force_download:
        print(f"{filename} already exists. Skipping download.")
    else:
        text = f"Downloading {truncate_string(filename, 50)}"
        with open(filepath, "wb") as file:
            total_size = int(response.headers.get("content-length", 0))
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=text,
                unit_divisor=1024,
                disable=not progress,
            ) as pb:
                if interrupt_check:
                    signal.signal(signal.SIGINT, lambda signum, frame: interrupt())
                for data in response.iter_content(block_size):
                    pb.update(len(data))
                    file.write(data)
    del response
    return filename

def factorize(num: int, max_value: int) -> list[float]:
    result = []
    while num > max_value:
        result.append(max_value)
        num /= max_value
    result.append(round(num, 4))
    return result

def upscale(
    img_list: list[PIL.Image.Image],
    model_name: str = "RealESRGAN_x4plus_anime_6B",
    scale_factor: float = 4,
    half_precision: bool = False,
    tile: int = 0,
    tile_pad: int = 10,
    pre_pad: int = 0,
) -> list[PIL.Image.Image]:
    # check model
    if model_name == "RealESRGAN_x4plus":
        upscale_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    elif model_name == "RealESRNet_x4plus":
        upscale_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=4,
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    elif model_name == "RealESRGAN_x4plus_anime_6B":
        upscale_model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4
        )
        netscale = 4
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    elif model_name == "RealESRGAN_x2plus":
        upscale_model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=2,
        )
        netscale = 2
        file_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    else:
        raise NotImplementedError("Model name not supported")

    # download model
    model_path = download_file(
        file_url, path="./upscaler-model", progress=False, interrupt_check=False
    )

    # declare the upscaler
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=os.path.join("./upscaler-model", model_path),
        dni_weight=None,
        model=upscale_model,
        tile=tile,
        tile_pad=tile_pad,
        pre_pad=pre_pad,
        half=half_precision,
        gpu_id=None,
    )

    # upscale
    torch.cuda.empty_cache()
    upscaled_imgs = []
    with tqdm(total=len(img_list)) as pb:
        for i, img in enumerate(img_list):
            img = np.array(img)
            outscale_list = factorize(scale_factor, netscale)
            with contextlib.redirect_stdout(StringIO()):
                for outscale in outscale_list:
                    curr_img = upsampler.enhance(img, outscale=outscale)[0]
                    img = curr_img
                upscaled_imgs.append(PIL.Image.fromarray(img))

            pb.update(1)
    torch.cuda.empty_cache()

    return upscaled_imgs

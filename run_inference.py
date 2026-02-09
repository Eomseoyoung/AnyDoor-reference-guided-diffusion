import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from cldm.hack import disable_verbosity, enable_sliced_attention
from datasets.data_utils import * 
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import albumentations as A
from omegaconf import OmegaConf
from PIL import Image


save_memory = False
disable_verbosity()
if save_memory:
    enable_sliced_attention()


config = OmegaConf.load('./configs/inference.yaml')
model_ckpt =  config.pretrained_model
model_config = config.config_file

model = create_model(model_config ).cpu()
model.load_state_dict(load_state_dict(model_ckpt, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)



def aug_data_mask(image, mask):
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ])
    transformed = transform(image=image.astype(np.uint8), mask = mask)
    transformed_image = transformed["image"]
    transformed_mask = transformed["mask"]
    return transformed_image, transformed_mask


def process_pairs(ref_image, ref_mask, tar_image, tar_mask):
    # ========= Reference ===========
    # ref expand 
    ref_box_yyxx = get_bbox_from_mask(ref_mask)

    # ref filter mask 
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)
    masked_ref_image = ref_image * ref_mask_3 + np.ones_like(ref_image) * 255 * (1-ref_mask_3)

    y1,y2,x1,x2 = ref_box_yyxx
    masked_ref_image = masked_ref_image[y1:y2,x1:x2,:]
    ref_mask = ref_mask[y1:y2,x1:x2]


    # 장애물 이미지 조절_1
    # ratio가 낮으면 객체 크게 높으면 객체 작게 
    ratio = np.random.randint(12, 13) / 10
    masked_ref_image, ref_mask = expand_image_mask(masked_ref_image, ref_mask, ratio=ratio)
    ref_mask_3 = np.stack([ref_mask,ref_mask,ref_mask],-1)

    # to square and resize
    masked_ref_image = pad_to_square(masked_ref_image, pad_value = 255, random = False)
    masked_ref_image = cv2.resize(masked_ref_image, (224,224) ).astype(np.uint8)

    ref_mask_3 = pad_to_square(ref_mask_3 * 255, pad_value = 0, random = False)
    ref_mask_3 = cv2.resize(ref_mask_3, (224,224) ).astype(np.uint8)
    ref_mask = ref_mask_3[:,:,0]

    # ref aug 
    masked_ref_image_aug = masked_ref_image #aug_data(masked_ref_image) 

    # collage aug 
    masked_ref_image_compose, ref_mask_compose = masked_ref_image, ref_mask #aug_data_mask(masked_ref_image, ref_mask) 
    masked_ref_image_aug = masked_ref_image_compose.copy()
    ref_mask_3 = np.stack([ref_mask_compose,ref_mask_compose,ref_mask_compose],-1)
    ref_image_collage = sobel(masked_ref_image_compose, ref_mask_compose/255)

    # 장애물 이미지 조절_2
    # ========= Target ===========
    tar_box_yyxx = get_bbox_from_mask(tar_mask)
    tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.1,1.2]) #기존
    #tar_box_yyxx = expand_bbox(tar_mask, tar_box_yyxx, ratio=[1.5, 1.8]) # 비율을 높이면 → 마스크 영역이 커져서 객체가 작게 배치
    
    
    
    # crop
    tar_box_yyxx_crop =  expand_bbox(tar_image, tar_box_yyxx, ratio=[1.5, 3])    #1.2 1.6
    tar_box_yyxx_crop = box2squre(tar_image, tar_box_yyxx_crop) # crop box
    y1,y2,x1,x2 = tar_box_yyxx_crop

    cropped_target_image = tar_image[y1:y2,x1:x2,:]
    tar_box_yyxx = box_in_box(tar_box_yyxx, tar_box_yyxx_crop)
    y1,y2,x1,x2 = tar_box_yyxx

    # collage
    ref_image_collage = cv2.resize(ref_image_collage, (x2-x1, y2-y1))
    ref_mask_compose = cv2.resize(ref_mask_compose.astype(np.uint8), (x2-x1, y2-y1))
    ref_mask_compose = (ref_mask_compose > 128).astype(np.uint8)

    collage = cropped_target_image.copy() 
    collage[y1:y2,x1:x2,:] = ref_image_collage

    collage_mask = cropped_target_image.copy() * 0.0
    collage_mask[y1:y2,x1:x2,:] = 1.0

    # the size before pad
    H1, W1 = collage.shape[0], collage.shape[1]
    cropped_target_image = pad_to_square(cropped_target_image, pad_value = 0, random = False).astype(np.uint8)
    collage = pad_to_square(collage, pad_value = 0, random = False).astype(np.uint8)
    collage_mask = pad_to_square(collage_mask, pad_value = -1, random = False).astype(np.uint8)

    # the size after pad
    H2, W2 = collage.shape[0], collage.shape[1]
    cropped_target_image = cv2.resize(cropped_target_image, (512,512)).astype(np.float32)
    collage = cv2.resize(collage, (512,512)).astype(np.float32)
    collage_mask  = (cv2.resize(collage_mask, (512,512)).astype(np.float32) > 0.5).astype(np.float32)

    masked_ref_image_aug = masked_ref_image_aug  / 255 
    cropped_target_image = cropped_target_image / 127.5 - 1.0
    collage = collage / 127.5 - 1.0 
    collage = np.concatenate([collage, collage_mask[:,:,:1]  ] , -1)

    item = dict(ref=masked_ref_image_aug.copy(), jpg=cropped_target_image.copy(), hint=collage.copy(), extra_sizes=np.array([H1, W1, H2, W2]), tar_box_yyxx_crop=np.array( tar_box_yyxx_crop ) ) 
    return item


def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 5 # maigin_pixel

    if W1 == H1:
        tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    return gen_image


def inference_single_image(ref_image, ref_mask, tar_image, tar_mask, guidance_scale = 5.0):
    item = process_pairs(ref_image, ref_mask, tar_image, tar_mask)
    ref = item['ref'] * 255
    tar = item['jpg'] * 127.5 + 127.5
    hint = item['hint'] * 127.5 + 127.5

    hint_image = hint[:,:,:-1]
    hint_mask = item['hint'][:,:,-1] * 255
    hint_mask = np.stack([hint_mask,hint_mask,hint_mask],-1)
    ref = cv2.resize(ref.astype(np.uint8), (512,512))

    seed = random.randint(0, 65535)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    ref = item['ref']
    tar = item['jpg'] 
    hint = item['hint']
    num_samples = 1

    control = torch.from_numpy(hint.copy()).float().cuda() 
    control = torch.stack([control for _ in range(num_samples)], dim=0)
    control = einops.rearrange(control, 'b h w c -> b c h w').clone()


    clip_input = torch.from_numpy(ref.copy()).float().cuda() 
    clip_input = torch.stack([clip_input for _ in range(num_samples)], dim=0)
    clip_input = einops.rearrange(clip_input, 'b h w c -> b c h w').clone()

    guess_mode = False
    H,W = 512,512

    cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning( clip_input )]}
    un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([torch.zeros((1,3,224,224))] * num_samples)]}
    shape = (4, H // 8, W // 8)

    if save_memory:
        model.low_vram_shift(is_diffusing=True)


    num_samples = 1 #gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
    image_resolution = 512  #gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
    strength = 1  #gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
    guess_mode = False #gr.Checkbox(label='Guess Mode', value=False)
    #detect_resolution = 512  #gr.Slider(label="Segmentation Resolution", minimum=128, maximum=1024, value=512, step=1)
    ddim_steps = 50 #gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
    scale = guidance_scale  #gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
    seed = -1  #gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
    eta = 0.0 #gr.Number(label="eta (DDIM)", value=0.0)

    model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
    samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)
    if save_memory:
        model.low_vram_shift(is_diffusing=False)

    x_samples = model.decode_first_stage(samples)
    x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy()#.clip(0, 255).astype(np.uint8)

    result = x_samples[0][:,:,::-1]
    result = np.clip(result,0,255)

    pred = x_samples[0]
    pred = np.clip(pred,0,255)[1:,:,:]
    sizes = item['extra_sizes']
    tar_box_yyxx_crop = item['tar_box_yyxx_crop'] 
    gen_image = crop_back(pred, tar_image, sizes, tar_box_yyxx_crop) 
    return gen_image


def load_and_resize_512(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"이미지 로드 실패: {path}")

    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    return img

if __name__ == '__main__':
    import os
    from pathlib import Path

    # ==== Example for inferring images from folders ===
    reference_folder = './anydoor_dataset/object/'  # 참조 이미지 폴더
    bg_image_folder = './anydoor_dataset/background/'  # 배경 이미지 폴더
    bg_mask_folder = './anydoor_dataset/mask/'  # 배경 마스크 폴더
    save_folder = './anydoor_dataset/result/'  # 결과 저장 폴더

    # ROI 설정 (True: ROI와 마스크 교집합 사용, False: 마스크 전체 사용)
    use_roi = True
    # ROI 영역 설정 (y1, y2, x1, x2) - 이미지 좌표 기준
    # 예시: 이미지 중앙 영역만 사용하려면 중앙 좌표를 계산하여 설정
    roi_ratio = {
        'y_start': 0.65,  # 이미지 높이의 25% 지점부터
        'y_end': 0.9,    # 이미지 높이의 75% 지점까지
        'x_start': 0.25,  # 이미지 너비의 25% 지점부터
        'x_end': 0.75     # 이미지 너비의 75% 지점까지
    }

    # 저장 폴더가 없으면 생성
    os.makedirs(save_folder, exist_ok=True)

    # 이미지 파일 확장자
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']

    # 참조 이미지 폴더에서 모든 이미지 파일 가져오기
    reference_images = [f for f in os.listdir(reference_folder)
                       if os.path.splitext(f)[1] in image_extensions]

    # 배경 이미지 폴더에서 모든 이미지 파일 가져오기
    bg_images = [f for f in os.listdir(bg_image_folder)
                if os.path.splitext(f)[1] in image_extensions]

    print(f"총 {len(reference_images)}개의 참조 이미지와 {len(bg_images)}개의 배경 이미지를 처리합니다.")
    print(f"총 생성될 이미지: {len(reference_images) * len(bg_images)}개\n")

    total_count = 0
    total_images = len(reference_images) * len(bg_images)

    # 각 참조 이미지에 대해 처리
    for ref_idx, ref_img_name in enumerate(reference_images, 1):
        reference_image_path = os.path.join(reference_folder, ref_img_name)
        ref_image_name = Path(ref_img_name).stem  # 확장자 제외한 이름

        try:
            # reference image + reference mask 로드
            image = load_and_resize_512(reference_image_path)
            mask = (image[:,:,-1] > 128).astype(np.uint8)
            image = image[:,:,:-1]
            image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
            ref_image = image
            ref_mask = mask

            # 각 배경 이미지에 대해 처리
            for bg_idx, bg_img_name in enumerate(bg_images, 1):
                total_count += 1
                bg_image_name = Path(bg_img_name).stem  # 확장자 제외한 이름

                # 배경 이미지 로드
                bg_image_path = os.path.join(bg_image_folder, bg_img_name)

                # 같은 이름의 마스크 파일 찾기 (확장자는 다를 수 있음)
                bg_mask_path = None
                for ext in image_extensions:
                    potential_mask = os.path.join(bg_mask_folder, bg_image_name + ext)
                    if os.path.exists(potential_mask):
                        bg_mask_path = potential_mask
                        break

                if bg_mask_path is None:
                    print(f"  [{total_count}/{total_images}] 마스크 파일을 찾을 수 없음: {bg_image_name}")
                    continue

                bg_mask_name = Path(bg_mask_path).stem

                print(f"[{total_count}/{total_images}] 처리 중: {ref_img_name} + {bg_img_name}")

                try:
                    # 배경 이미지 로드
                    back_image = cv2.imread(bg_image_path).astype(np.uint8)
                    back_image = cv2.cvtColor(back_image, cv2.COLOR_BGR2RGB)

                    # 배경 마스크 로드
                    tar_mask = cv2.imread(bg_mask_path)[:,:,0] > 128
                    tar_mask = tar_mask.astype(np.uint8)

                    # ROI와 마스크 교집합 계산
                    if use_roi:
                        h, w = tar_mask.shape
                        y1 = int(h * roi_ratio['y_start'])
                        y2 = int(h * roi_ratio['y_end'])
                        x1 = int(w * roi_ratio['x_start'])
                        x2 = int(w * roi_ratio['x_end'])

                        # ROI 마스크 생성
                        roi_mask = np.zeros_like(tar_mask)
                        roi_mask[y1:y2, x1:x2] = 1

                        # 원본 마스크와 ROI의 교집합
                        tar_mask = tar_mask & roi_mask

                        print(f"  → ROI 적용: ({y1}:{y2}, {x1}:{x2})")

                    # 추론 실행
                    gen_image = inference_single_image(ref_image.copy(), ref_mask.copy(),
                                                      back_image.copy(), tar_mask.copy())

                    # 저장 경로: 참조이미지이름_마스크이름_배경이미지이름.png
                    save_filename = f"{ref_image_name}_{bg_mask_name}_{bg_image_name}.png"
                    save_path = os.path.join(save_folder, save_filename)

                    cv2.imwrite(save_path, gen_image[:,:,::-1])
                    print(f"  → 저장 완료: {save_filename}")

                except Exception as e:
                    print(f"  → 오류 발생: {e}")
                    continue

        except Exception as e:
            print(f"참조 이미지 로드 실패 ({ref_img_name}): {e}")
            continue

    print(f"\n모든 이미지 처리 완료! (총 {total_count}개 처리)")

    '''
    # ==== Example for inferring VITON-HD Test dataset ===

    from omegaconf import OmegaConf
    import os 
    DConf = OmegaConf.load('./configs/datasets.yaml')
    save_dir = './VITONGEN'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    test_dir = DConf.Test.VitonHDTest.image_dir
    image_names = os.listdir(test_dir)
    
    for image_name in image_names:
        ref_image_path = os.path.join(test_dir, image_name)
        tar_image_path = ref_image_path.replace('/cloth/', '/image/')
        ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
        tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        gt_image = cv2.imread(tar_image_path)
        gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 5

        gen_image = inference_single_image(ref_image, ref_mask, gt_image.copy(), tar_mask)
        gen_path = os.path.join(save_dir, image_name)

        vis_image = cv2.hconcat([ref_image, gt_image, gen_image])
        cv2.imwrite(gen_path, vis_image[:,:,::-1])
    '''

    


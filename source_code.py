import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io

def remove_background(person_path):
    session = new_session("u2net", providers=['CPUExecutionProvider'])
    with open(person_path, 'rb') as f:
        input_image = f.read()
    output_image = remove(input_image, session=session)
    return Image.open(io.BytesIO(output_image)).convert("RGBA")

def refine_edges(person_img, erosion_size=3):
    alpha = np.array(person_img.split()[-1])
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_alpha = cv2.erode(alpha, kernel, iterations=1)
    eroded_alpha_pil = Image.fromarray(eroded_alpha)
    blurred_alpha = eroded_alpha_pil.filter(ImageFilter.GaussianBlur(radius=2))
    refined_person_img = person_img.copy()
    refined_person_img.putalpha(blurred_alpha)
    return refined_person_img

def resize_person_dynamic(person_img, bg_img, height_ratio=0.6):
    bg_w, bg_h = bg_img.size
    target_height = int(bg_h * height_ratio)
    if person_img.height == 0: return person_img
    aspect_ratio = person_img.width / person_img.height
    target_width = int(target_height * aspect_ratio)
    return person_img.resize((target_width, target_height), Image.LANCZOS)

def color_match_luminance(person_img, bg_img, strength=0.7):
    alpha = person_img.split()[-1]
    mask_np = np.array(alpha)
    person_np_rgb = np.array(person_img.convert("RGB"))
    person_lab = cv2.cvtColor(person_np_rgb, cv2.COLOR_RGB2LAB)
    bg_lab = cv2.cvtColor(np.array(bg_img.convert("RGB")), cv2.COLOR_RGB2LAB)
    p_l, p_a, p_b = cv2.split(person_lab)
    b_l, _, _ = cv2.split(bg_lab)
    p_l_mean, p_l_std = cv2.meanStdDev(p_l, mask=mask_np)
    b_l_mean, b_l_std = cv2.meanStdDev(b_l)
    if p_l_std[0][0] < 1e-5: p_l_std[0][0] = 1e-5
    corrected_l = (p_l.astype(np.float32) - p_l_mean) / p_l_std * b_l_std + b_l_mean
    blended_l = (corrected_l * strength) + (p_l.astype(np.float32) * (1 - strength))
    blended_l = np.clip(blended_l, 0, 255).astype(np.uint8)
    final_lab = cv2.merge([blended_l, p_a, p_b])
    final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_LAB2RGB)
    person_adjusted_img = Image.fromarray(final_rgb, 'RGB')
    person_adjusted_img.putalpha(alpha)
    return person_adjusted_img

def add_light_wrap(fg_img, bg_img, strength=0.4, blur_radius=15):
    fg_np = np.array(fg_img)
    alpha = fg_np[..., 3] / 255.0
    inv_alpha = 1 - alpha
    bg_crop = bg_img.crop((0, 0, fg_img.width, fg_img.height))
    blurred_bg = bg_crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    blurred_np = np.array(blurred_bg)
    wrap = ((blurred_np[..., :3] - fg_np[..., :3]) * inv_alpha[..., None] * strength).astype(np.int16)
    new_rgb = np.clip(fg_np[..., :3] + wrap, 0, 255).astype(np.uint8)
    result_np = np.dstack((new_rgb, fg_np[..., 3]))
    return Image.fromarray(result_np, 'RGBA')

def add_ground_shadow(bg_img, person_pos, person_size, shadow_size=(1.1, 0.25), blur_radius=18, opacity=0.3):
    shadow_w = int(person_size[0] * shadow_size[0])
    shadow_h = int(person_size[1] * shadow_size[1])
    shadow = Image.new("RGBA", (shadow_w, shadow_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.ellipse([0, 0, shadow_w, shadow_h], fill=(0, 0, 0, int(255 * opacity)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
    shadow_x = person_pos[0] + (person_size[0] - shadow_w) // 2
    shadow_y = person_pos[1] + person_size[1] - shadow_h // 2
    result = bg_img.copy()
    result.paste(shadow, (shadow_x, shadow_y), shadow)
    return result

def composite(person_path, bg_path, output_path, color_strength=0.7):
    bg_img = Image.open(bg_path).convert("RGBA")
    person_img_raw = remove_background(person_path)
    person_refined = refine_edges(person_img_raw)
    person_resized = resize_person_dynamic(person_refined, bg_img)
    if person_resized.width == 0 or person_resized.height == 0:
        print("❌ Error: Person image is empty after resizing.")
        return
    person_final = color_match_luminance(person_resized, bg_img, strength=color_strength)

    # Reduce brightness slightly for better background integration
    enhancer = ImageEnhance.Brightness(person_final.convert("RGB"))
    slightly_darker_rgb = enhancer.enhance(0.65) 
    person_final = Image.merge("RGBA", (*slightly_darker_rgb.split(), person_final.split()[-1]))

    person_final = add_light_wrap(person_final, bg_img, strength=0.4, blur_radius=12)
    bg_w, bg_h = bg_img.size
    p_w, p_h = person_final.size
    position = ((bg_w - p_w) // 2, bg_h - p_h - 20)
    bg_with_shadow = add_ground_shadow(bg_img, position, person_final.size)
    final_image = bg_with_shadow.copy()
    final_image.paste(person_final, position, person_final)
    final_image.convert("RGB").save(output_path, 'PNG', quality=95)
    print(f"✅ Final, color-accurate composite saved to: {output_path}")

# --- Example Usage ---
if __name__ == '__main__':
    try:
        composite(
            person_path='person1.jpg',
            bg_path='background5.jpg',
            output_path='final_photorealistic_result.png',
            color_strength=0.7
        )
    except FileNotFoundError:
        print("Error: Make sure 'person1.jpg' and 'background4.jpg' exist in the script's directory.")





















import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

def calculate_psnr(img1, img2):
    return psnr(img1, img2)

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

def calculate_metrics(img1, img2):
    if img1.shape != img2.shape:
    # Resize img2 to match the shape of img1
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    
    # Convert images to grayscale if needed
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR
    psnr_value = calculate_psnr(img1, img2)
   
    # Calculate SSIM
    ssim_value = calculate_ssim(img1_gray, img2_gray)
    return psnr_value, ssim_value


def main():
    # Load the images
    img1 = cv2.imread('../Inference/inputs/highres11.jpg')  
    img2 = cv2.imread('../Inference/8R_highres11.jpg')
    img3 = cv2.imread('../Inference/16R_highres11.jpg')
    img4 = cv2.imread('../Inference/28R_highres11.jpg')  

    psnr_value, ssim_value = calculate_metrics(img1, img2)
    print('8 Channels')
    print(f'PSNR: {psnr_value} dB')
    print(f'SSIM: {ssim_value}')

    psnr_value, ssim_value = calculate_metrics(img1, img3)
    print('16 Channels')
    print(f'PSNR: {psnr_value} dB')
    print(f'SSIM: {ssim_value}')

    psnr_value, ssim_value = calculate_metrics(img1, img4)
    print('28 Channels')
    print(f'PSNR: {psnr_value} dB')
    print(f'SSIM: {ssim_value}')

main()



# Baboon 

# torch.Size([8, 30, 31])
# torch.Size([3, 120, 125])



# Landscape

# torch.Size([8, 120, 250])
# torch.Size([3, 482, 1000])

# torch.Size([16, 120, 250])
# torch.Size([3, 482, 1000])

# torch.Size([28, 120, 250])
# torch.Size([3, 482, 1000])

# Hari Chipkali

# torch.Size([28, 166, 250])
# torch.Size([3, 667, 1000])
# PSNR: 24.035156180758243 dB
# SSIM: 0.8354448440651789

# torch.Size([16, 166, 250])
# torch.Size([3, 667, 1000])
# PSNR: 23.688985383271543 dB
# SSIM: 0.8067037859876772

# torch.Size([8, 166, 250])
# torch.Size([3, 667, 1000])
# PSNR: 21.919089837802936 dB
# SSIM: 0.7809019191751623

# Sea

# torch.Size([8, 300, 480])
# torch.Size([3, 1200, 1920])

# Ishowspeed

# torch.Size([28, 103, 155])
# torch.Size([3, 412, 620])

# Beach

# torch.Size([8, 125, 200])
# torch.Size([3, 500, 800])




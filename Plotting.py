import matplotlib.animation as animation
import matplotlib.image as mpimg
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import cv2
import copy
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

num_images_to_show = 2

f, axarr = plt.subplots(num_images_to_show, 4)

axarr[0,0].title.set_text('Original \n Image')
axarr[0,1].title.set_text('Reconstructed Image with \n 43% Compression')
axarr[0,2].title.set_text('Reconstructed Image with \n 68% Compression')
axarr[0,3].title.set_text('Reconstructed Image with \n 84% Compression')

for i in range(4):
    axarr[0,i].title.set_fontsize(15)

to_tensor = transforms.ToTensor()


def ShowGraph(i, img, img_8, img_16, img_28):
    # Convert images to float
    Img = copy.deepcopy(img)
    Img_8 = copy.deepcopy(img_8)
    Img_16 = copy.deepcopy(img_16)
    Img_28 = copy.deepcopy(img_28)
    
    def calps(img1, img2):
        numpy_image = np.array(img1)
        img1 = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)  

        numpy_image = np.array(img2)
        img2 = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        psnr_value = psnr(img1, img2)

        ssim_value = ssim(img1_gray, img2_gray)
        return psnr_value, ssim_value


    img = to_tensor(img)
    img_8 = to_tensor(img_8)
    img_16 = to_tensor(img_16)
    img_28 = to_tensor(img_28)

    psnr_8, ssim_8 = calps(Img, Img_8)
    psnr_16, ssim_16 = calps(Img, Img_16)
    psnr_28, ssim_28 = calps(Img, Img_28)

    # Calculate PSNR and SSIM for each reconstructed image
    # psnr_8, ssim_8 = psnr(Img, Img_8), ssim(Img, Img_8, multichannel=True)
    # psnr_16, ssim_16 = psnr(Img, Img_16), ssim(Img, Img_16, multichannel=True)
    # psnr_28, ssim_28 = psnr(Img, Img_28), ssim(Img, Img_28, multichannel=True)

    # Display images with PSNR and SSIM values in the title
    axarr[i,0].imshow((img.permute(1, 2, 0) * 0.5) + 0.5)
    axarr[i,1].imshow((img_28.permute(1, 2, 0) *0.5) + 0.5)
    axarr[i,1].title.set_text(f' 43% Compression\nPSNR: {psnr_28:.2f}, SSIM: {ssim_28:.2f}')
    axarr[i,2].imshow((img_16.permute(1, 2, 0) *0.5) + 0.5)
    axarr[i,2].title.set_text(f' 68% Compression\nPSNR: {psnr_16:.2f}, SSIM: {ssim_16:.2f}')
    axarr[i,3].imshow((img_8.permute(1, 2, 0) *0.5) + 0.5)
    axarr[i,3].title.set_text(f' 84% Compression\nPSNR: {psnr_8:.2f}, SSIM: {ssim_8:.2f}')

    f.set_figheight(20)
    f.set_figwidth(20)


img = Image.open('inputs/highres2.jpg')
img_8 = Image.open('8R_highres2.jpg')
img_16 = Image.open('16R_highres2.jpg')
img_28 = Image.open('28R_highres2.jpg')

Img = Image.open('inputs/highres4.jpg')
Img_8 = Image.open('8R_highres4.jpg')
Img_16 = Image.open('16R_highres4.jpg')
Img_28 = Image.open('28R_highres4.jpg')

ShowGraph(0, img, img_8, img_16, img_28)
ShowGraph(1, Img, Img_8, Img_16, Img_28)

plt.show()
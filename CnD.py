def CnD():
    import torch
    import numpy as np
    from torchvision import transforms
    import torchvision.utils
    from PIL import Image
    import EncoderGenerator
    import cv2
    from skimage.metrics import structural_similarity as ssim

    num_channels_in_encoder = 28

    # Create instances of the models
    netE = EncoderGenerator.Encoder()
    netG = EncoderGenerator.Generator()

    # Load the saved state dictionaries
    netE.load_state_dict(torch.load("models/netE" + str(num_channels_in_encoder) + ".model", map_location=torch.device('cpu')))
    netG.load_state_dict(torch.load("models/netG" + str(num_channels_in_encoder) + ".model", map_location=torch.device('cpu')))

    image_dir = "./inputs/"
    image_name = "highres9.jpg"

    img = Image.open(image_dir + image_name)
    
    # Apply similar preprocessing
    preprocess = transforms.Compose([
        #transforms.CenterCrop((218, 178)),  # Adjust the cropping size if needed
        transforms.ToTensor(),
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Use the same normalization as during training
    ])

    preprocess_img = preprocess(img)

    netE.eval()
    netG.eval()

    # Encode the test image
    with torch.no_grad():
        encoded_img = netE(preprocess_img)  # Assuming img is a tensor

    # Decode the encoded representation
    with torch.no_grad():
        reconstructed_img = netG(encoded_img)

        denormalize = transforms.Compose([
        #transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        transforms.ToPILImage()
    ])
    
    #print(encoded_img.cpu().type())
    #print(reconstructed_img.cpu().type())
    print(encoded_img.size())
    print(preprocess_img.size())

    # # Load the image using PIL
    # image_path = 'path/to/your/image.jpg'
    # img = Image.open(image_path)

    # # Define a transformation to convert the image to a PyTorch tensor
    # transform = transforms.ToTensor()

    # # Apply the transformation to the image
    # img_tensor = transform(img)

    # # Print the shape of the resulting tensor
    # print("Shape of the tensor:", img_tensor.shape

    # reconstructed_img_denorm = denormalize(reconstructed_img.squeeze().cpu())
    # reconstructed_img_denorm.save("reconstructed_" + image_name)
    
    torchvision.utils.save_image(reconstructed_img, str(num_channels_in_encoder) + 'R_' + image_name)
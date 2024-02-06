def EnG():
    import torch
    import numpy as np
    from torchvision import transforms
    import torchvision.utils
    from PIL import Image
    import os
    import EncoderGenerator

    num_channels_in_encoder = 8

    # Define the device
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")

    # print(device)
    # print(torch.cuda.get_device_name(torch.cuda.current_device()))

    try:
        torch.cuda.empty_cache()
    except: 
        torch.cpu.empty_cache()

    # Create instances of the models
    netE = EncoderGenerator.Encoder()
    netG = EncoderGenerator.Generator()

    netE.to(device)
    netG.to(device)

    # Load the saved state dictionaries
    netE.load_state_dict(torch.load("../models/netE" + str(num_channels_in_encoder) + ".model", map_location=device))
    netG.load_state_dict(torch.load("../models/netG" + str(num_channels_in_encoder) + ".model", map_location=device))

    image_dir = "./inputs/"
    image_name = "highres.jpg"

    img = Image.open(image_dir + image_name)
    
    # Apply similar preprocessing
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    preprocess_img = preprocess(img).unsqueeze(0).to(device)

    netE.eval()
    netG.eval()

    try:
        torch.cuda.empty_cache()
    except: 
        torch.cpu.empty_cache()

    # Encode the test image
    with torch.no_grad():
        encoded_img = netE(preprocess_img)  # Assuming img is a tensor
    
    del preprocess_img
    del netE

    try:
        torch.cuda.empty_cache()
    except: 
        torch.cpu.empty_cache()

    # Decode the encoded representation
    with torch.no_grad():
        reconstructed_img = netG(encoded_img)
        denormalize = transforms.Compose([
        transforms.ToPILImage() 
        ])
    

    reconstructed_img_denorm = denormalize(reconstructed_img.squeeze().cpu())

    output_folder = str(num_channels_in_encoder) + "_Channels/"

    os.makedirs(output_folder, exist_ok=True)

    # Save the reconstructed image with the desired filename
    reconstructed_img_denorm.save(os.path.join(output_folder, str(num_channels_in_encoder) + 'R_' + image_name))
    
    # torchvision.utils.save_image(reconstructed_img, str(num_channels_in_encoder) + "_Channels/" + str(num_channels_in_encoder) + 'R_' + image_name)
    try:
        torch.cuda.empty_cache()
    except: 
        torch.cpu.empty_cache()
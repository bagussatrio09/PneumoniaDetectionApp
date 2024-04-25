import torch
import glob as glob
import cv2

import torchvision
from torchvision import transforms
from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from PIL import Image
import os
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
matplotlib.use('agg')
import io
import base64
import time

def checking_file_format(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png','zip'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file(filename):
    return filename.lower().endswith(('.jpg', '.jpeg', '.png'))


def detection(image_folder='static/files/'):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print(device)


    class PneumoniaDataset(object):
        def __init__(self, transforms, path):
            self.transforms = transforms
            self.path = path
            self.imgs = list(sorted(os.listdir(self.path)))

        def __getitem__(self, idx):
            file_image = self.imgs[idx]
            img_path = os.path.join(self.path, file_image)

            img = Image.open(img_path).convert("RGB")

            if self.transforms is not None:
                img = self.transforms(img)

            return img

        def __len__(self):
            return len(self.imgs)

    data_transform = transforms.Compose([transforms.ToTensor()])

    def collate_fn(batch):
        return torch.stack(batch)

    test_dataset = PneumoniaDataset(data_transform, image_folder)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)


    def create_model(num_classes):
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1
        )
        num_anchors = model.head.classification_head.num_anchors

        model.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        return model
    
    model = create_model(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    
    checkpoint = torch.load('best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    plot_urls_list = []  # List to store plot URLs for all detected images
    detection_statuses_list = []  # List to store detection statuses for all images

    
    def plot_image_from_output(img, annotation):
        detectionStatus = False
        img = img.cpu().permute(1,2,0)

        _,ax = plt.subplots(1)
        ax.imshow(img)

        for idx in range(len(annotation["boxes"])):
            xmin, ymin, xmax, ymax = annotation["boxes"][idx]

            if annotation['labels'][idx] == 1:
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none')
                label = "Pneumonia"
                detectionStatus = True
            else:
                label = "False"

            ax.add_patch(rect)

            # Menentukan koordinat untuk teks di luar kotak
            text_x = xmin + 20
            text_y = ymin + 5

            # Menambahkan label teks di luar kotak
            score = annotation['scores'][idx] if 'scores' in annotation else None
            text = f"{label} {int(score * 100)}%" if score is not None else label
            ax.text(text_x, text_y, text, fontsize=10, color='white', verticalalignment='center', bbox={'color': 'black', 'alpha': 0.7, 'pad': 0})

        # plt.show()
        imgPng = io.BytesIO()
        plt.savefig(imgPng, format='png')
        imgPng.seek(0)
        plot_url = base64.b64encode(imgPng.getvalue()).decode()
        return plot_url, detectionStatus

    def make_prediction(model, img, threshold):
        model.eval()
        preds = model(img)
        for i in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[i]['scores']) :
                if score > threshold :
                    idx_list.append(idx)

            preds[i]['boxes'] = preds[i]['boxes'][idx_list]
            preds[i]['scores'] = preds[i]['scores'][idx_list]
            preds[i]['labels'] = preds[i]['labels'][idx_list]

        return preds

    with torch.no_grad():
        start_time = time.time() 
        for imgs in test_loader:
            imgs = list(img.to(device) for img in imgs)
            pred = make_prediction(model, imgs, 0.3)

            _idx = 0  # Choose an index based on your use case

            for key in pred[_idx]:
                pred[_idx][key] = pred[_idx][key].cpu()

            plot_urls, detection_status = plot_image_from_output(imgs[_idx], pred[_idx])

            plot_urls_list.append(plot_urls)
            detection_statuses_list.append(detection_status)
        end_time = time.time()  # Waktu selesai deteksi
        total_time = end_time - start_time

    return plot_urls_list, detection_statuses_list, total_time
    
# def plot_image(image_path):
    img = Image.open(image_path)

    fig,ax = plt.subplots(1, )
    ax.imshow(img, cmap='gray')

    imgPng = io.BytesIO()
    plt.savefig(imgPng, format='png')
    imgPng.seek(0)
    plot_url = base64.b64encode(imgPng.getvalue()).decode()
    return plot_url

def plot_images_in_folder(folder_path):
    plot_urls = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            
            # Check if the file is an allowed image file
            if not allowed_file(file):
                continue
            
            try:
                # Process only valid image files
                img = Image.open(image_path)

                # Convert the image to grayscale
                img_gray = img.convert('L')

                fig, ax = plt.subplots(1)

                # Plot the grayscale image
                ax.imshow(img_gray, cmap='gray')
                file_name = os.path.splitext(file)[0]
                ax.set_title(f'Image: {file_name}')

                imgPng = io.BytesIO()
                plt.savefig(imgPng, format='png')
                imgPng.seek(0)
                plot_url = base64.b64encode(imgPng.getvalue()).decode()
                plot_urls.append(plot_url)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    return plot_urls

def delete_file(filepath):
    try:
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print("Error deleting file:", e)


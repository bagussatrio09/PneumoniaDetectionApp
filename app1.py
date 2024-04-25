from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import zipfile
from PIL import Image
from utils1 import detection, delete_file, checking_file_format , plot_images_in_folder
import os
app = Flask(__name__)

app.config['SECRET_KEY'] = '124-407-823'
app.config['UPLOAD_FOLDER'] = 'static/files/'

@app.route('/')
def index():
    delete_file(app.config['UPLOAD_FOLDER'])
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    delete_file(app.config['UPLOAD_FOLDER'])

    uploaded_file = request.files.get('image')
    if not uploaded_file or not checking_file_format(uploaded_file.filename):
        return render_template('index.html')

    # Save the uploaded file
    uploaded_filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
    uploaded_file.save(uploaded_filepath)

    # Check if it's a ZIP file
    if uploaded_file.filename.endswith('.zip'):
        # Extract images from the ZIP file
        extracted_folder = os.path.join(app.config['UPLOAD_FOLDER'])
        with zipfile.ZipFile(uploaded_filepath, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder)

        # Perform operations on extracted images
        plot_imgs = plot_images_in_folder(extracted_folder)

        # Remove the ZIP file after extraction
        os.remove(uploaded_filepath)

        return render_template('index.html', plot_imgs=plot_imgs, alert="File berhasil diupload dan diekstrak")

    else:
        extracted_folder = os.path.join(app.config['UPLOAD_FOLDER'])
        # Perform operations on the single uploaded image (png/jpg)
        plot_imgs = plot_images_in_folder(extracted_folder)

        return render_template('index.html', plot_imgs=plot_imgs, alert="File berhasil diupload")


@app.route('/process', methods=['POST'])
def process():
    upload_folder = app.config['UPLOAD_FOLDER']
    
    plot_imgs = plot_images_in_folder(upload_folder)
    plot_urls_list, detection_statuses_list, total_time = detection(image_folder=upload_folder)
    
    detection_time = f"{total_time:.2f}"

    if plot_urls_list:
        return render_template('index.html', plot_imgs=plot_imgs, plot_urls_list=plot_urls_list, detection_statuses_list=detection_statuses_list, detection_time=detection_time, alert="Proses selesai")
    else:
        return render_template('index.html', alert="Tidak ada file yang dipilih atau file tidak valid")


if __name__ == '__main__':
    app.debug = True
    app.run(debug=True, port=8000, ) 
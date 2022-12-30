import cv2
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from datetime import datetime

############################# start - variables ################################
sr = cv2.dnn_superres.DnnSuperResImpl_create()

models_2x = ['EDSR_x2.pb']
models_3x = ['a']
models_4x = ['EDSR_x4.pb', 'LapSRN_x4.pb']
models_8x = ['a']

BASE_PATH = 'models/'


############################# start - functions ################################

def upscale(model_path: str, model_name: str, scale: str, img, img_type: str):
    scale = int(scale.split('x')[0])
    sr.readModel(model_path)
    sr.setModel(model_name, scale)
    result = sr.upsample(img)
    img_type = img_type.split('/')[1]
    save_path = f'result.{img_type}'
    plt.imsave(save_path, result[:, :, ::-1])
    return result[:, :, ::-1], save_path


def get_modelname(selected_model: str) -> str:
    if 'EDSR' in selected_model:
        return 'edsr'
    elif 'LapSRN' in selected_model:
        return 'lapsrn'


def model_selector(scale: str) -> str:
    model = ''
    if scale == '2x':
        model = st.selectbox(
            'Which model do you want to use?',
            ('Not selected', models_2x[0], models_2x[0]))
    elif scale == '3x':
        model = st.selectbox(
            'Which model do you want to use?',
            ('Not selected', models_3x[0], models_3x[0]))
    elif scale == '4x':
        model = st.selectbox(
            'Which model do you want to use?',
            ('Not selected', models_4x[0], models_4x[1]))
    elif scale == '8x':
        model = st.selectbox(
            'Which model do you want to use?',
            ('Not selected', models_8x[0], models_8x[0]))
    else:
        return False, False

    model_name = get_modelname(model)
    return model, model_name


############################# start - Streamlit ################################

st.title('Free image upscaler using deep learning')
st.markdown(
    'By [Mehrdad Mohammadian](https://mehrdad-dev.github.io)', unsafe_allow_html=True)

about = """
This demo provides a simple interface to upscale your images using deep learning. 
In streamlit, there is some shortages in terms of CPU, to solve this issue use codes in GitHub on your own device.
"""
st.markdown(about, unsafe_allow_html=True)

scale = st.selectbox(
    'Which scale do you want to apply to on your image?',
    ('Not selected', '2x', '3x', '4x', '8x'))


uploaded_file = None
model, model_name = model_selector(scale)
if model and model != 'Not selected':
    model_path = BASE_PATH + scale + '/' + model
    uploaded_file = st.file_uploader("Upload a jpg image", type=["jpg", "png"])


image = None
if uploaded_file is not None:
    # file_details = {"Filename":uploaded_file.name,"FileType":uploaded_file.type,"FileSize":uploaded_file.size}
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption='Your uploaded image')

    left_column, right_column = st.columns(2)
    pressed = left_column.button('Upscale!')

    if pressed:
        pressed = False
        st.info('Processing ...')
        result, save_path = upscale(
            model_path, model_name, scale, image, uploaded_file.type)
        st.success('Image is ready, you can download it!')
        st.balloons()
        st.image(result, channels="RGB", caption='Your upscaled image')
        with open(save_path, 'rb') as f:
            st.download_button('Download the image', f, file_name=scale + '_' + str(datetime.now()) + '_' + save_path)

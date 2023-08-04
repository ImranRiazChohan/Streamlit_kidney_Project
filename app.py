# Python In-built packages
from pathlib import Path
import PIL
from ultralytics import YOLO
# External packages
import streamlit as st
from skimage import measure
import torchvision as T
import matplotlib.pyplot as plt

# Local Modules
import settings
import helper

# Setting page layout
st.set_page_config(
    page_title="Segmentation using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Overview','Segmentation'])

#***********************************************************************
# Selecting OverView
if model_type == 'Overview':
    st.title('OverView About Kidney Tumor Segmentation')




#***********************************************************************

# Selection Segmentation
elif model_type == 'Segmentation':

    st.title("Kidney Stone and Tumor Segmentation using YOLOv8")
    model_path = Path(settings.SEGMENTATION_MODEL)

    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)

    # Upload Image
    st.sidebar.header("Uplaod Image")
    image_path=st.sidebar.file_uploader('Choose an image...',type=['jpg','jpeg','png'])
    
    col1, col2 = st.columns(2)

    if image_path is not None:           
        # Orignal Image
        with col1:
            uploaded_image = PIL.Image.open(image_path)
            st.image(image_path, caption="Uploaded Image",use_column_width=True)
        
        if st.sidebar.button('Apply Segmentation Model'):
            results = list(model(uploaded_image))
            result=results[0]
            no_of_masks,x,y=result.masks.shape
            # Segment Image    
            with col2:
                res_plotted = result.plot(line_width=1,labels=True)[:, :, ::-1]

                st.image(res_plotted, caption='Segmented Image',use_column_width=True)
            tab1,tab2=st.tabs(['Kidney','Tumor'])
            #mask properties and images
            for i in range(no_of_masks):
                    if i<2:
                        with tab1:
                            mask_image=(result.masks.data[i].numpy()*255).astype('uint8')
                            label = measure.label(mask_image) # same image_binary as above
                            props = measure.regionprops(label)
                            for label in props:
                                area=label.area
                                perimeter=label.perimeter
                            st.image(mask_image,caption=f'Kidney Area:{area} , Perimeter:{round(perimeter,2)}',width=300)
                    else:
                        with tab2:
                            mask_image=(result.masks.data[i].numpy()*255).astype('uint8')
                            label = measure.label(mask_image) # same image_binary as above
                            props = measure.regionprops(label)
                            for label in props:
                                area=label.area
                                perimeter=label.perimeter
                            st.image(mask_image,caption=f'Tumor Area:{area} , Perimeter:{round(perimeter,2)}',width=300)

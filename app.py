import streamlit as st
from PIL import Image, ImageFilter
from try_on_clothes.script import predict
import time

from human_parsing.evaluate_human_parsing import execute
from pose_map.pose_parser import pose_parse
from cloth_mask.evaluate_mask import execute_mask

st.title("Virtual Try ON")

# Preload images
cloths = {
    '001247': Image.open('./Database/val/cloth/001247_1.jpg'),
    '001401': Image.open('./Database/val/cloth/001401_1.jpg'),
    '001500': Image.open('./Database/val/cloth/001500_1.jpg'),
    '001719': Image.open('./Database/val/cloth/001719_1.jpg'),
    '002061': Image.open('./Database/val/cloth/002061_1.jpg'),
    '002337': Image.open('./Database/val/cloth/002337_1.jpg'),
    '002385': Image.open('./Database/val/cloth/002385_1.jpg'),
    '002599': Image.open('./Database/val/cloth/002599_1.jpg'),
    '003086': Image.open('./Database/val/cloth/003086_1.jpg'),
    '006158': Image.open('./Database/val/cloth/006158_1.jpg'),
    '006159': Image.open('./Database/val/cloth/006159_1.jpg')
}

# Initialize session state
if 'option_selected' not in st.session_state:
    st.session_state.option_selected = None
if 'uploaded_cloth' not in st.session_state:
    st.session_state.uploaded_cloth = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''
if 'selected_cloth' not in st.session_state:
    st.session_state.selected_cloth = ''

# File uploader for person image
uploaded_person = st.file_uploader("Upload a Photo", type=["jpg", "jpeg", "png"], key='uploaded_person')

if uploaded_person is not None:
    st.write("Select options")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Upload shirt"):
            st.session_state.option_selected = 'upload_shirt'
    with col2:
        if st.button("Select item in library"):
            st.session_state.option_selected = 'select_item'

    if st.session_state.option_selected == 'upload_shirt':
        uploaded_cloth = st.file_uploader("Upload a Shirt", type=["jpg", "jpeg", "png"], key='uploaded_cloth')
        user_input = st.text_input("Enter the User Name... eg sourav", key='user_input')
        selected = "new_cloth"
        if user_input and uploaded_cloth:
            person = Image.open(uploaded_person)
            st.image(person, caption=user_input, width=100, use_column_width=False)
            st.write("Saving Image")
            bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.09)
                bar.progress(percent_complete + 1)
            person.save("./Database/val/person/" + user_input + ".jpg")
            cloth = Image.open(uploaded_cloth)
            cloth = cloth.resize((192, 256))
            st.image(cloth, caption="new_cloth_1", width=100, use_column_width=False)
            st.write("Saving Image")
            bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.09)
                bar.progress(percent_complete + 1)
            cloth.save("./Database/val/cloth/" + "new_cloth_1" + ".jpg")
            execute_mask()
            progress_bar = st.progress(0)
            st.write("Generating Mask and Pose Pairs")
            pose_parse(user_input)
            execute()
            for percent_complete in range(100):
                time.sleep(0.05)
                progress_bar.progress(percent_complete + 1)
            st.write("Please click the Click Button after Pose pairs and Masks are generated")
            if st.button('Execute'):
                with open("./Database/val_pairs.txt", "w") as f:
                    f.write(user_input + ".jpg " + selected + "_1.jpg")
                predict()
                im = Image.open("./output/second/TOM/val/" + selected + "_1.jpg")
                width, height = im.size
                left = width / 3
                top = 2 * height / 3
                right = 2 * width / 3
                bottom = height
                im1 = im.crop((left, top, right, bottom))
                im1 = im1.resize((256, 256))
                im1 = im1.filter(ImageFilter.SHARPEN)
                im1.save("./output/second/TOM/val/" + selected + "_1.jpg")
                execute_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.08)
                    execute_bar.progress(percent_complete + 1)
                result = Image.open("./output/second/TOM/val/" + selected + "_1.jpg")
                st.image(result, caption="Result", width=200, use_column_width=False)

        else:
            st.error("Please upload a shirt and enter the user name.")

    elif st.session_state.option_selected == 'select_item':
        for key, cloth in cloths.items():
            st.sidebar.image(cloth, caption=key, width=100, use_column_width=False)
        user_input = st.text_input("Enter the User Name... eg sourav", key='user_input')
        selected = st.selectbox('Select the Item Id:', [''] + list(cloths.keys()), format_func=lambda x: 'Select an option' if x == '' else x, key='selected_cloth')

        if user_input and selected:
            person = Image.open(uploaded_person)
            st.image(person, caption=user_input, width=100, use_column_width=False)
            st.write("Saving Image")
            bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.09)
                bar.progress(percent_complete + 1)
            person.save("./Database/val/person/" + user_input + ".jpg")
            progress_bar = st.progress(0)
            st.write("Generating Mask and Pose Pairs")
            pose_parse(user_input)
            execute()
            for percent_complete in range(100):
                time.sleep(0.05)
                progress_bar.progress(percent_complete + 1)
            st.write("Please click the Click Button after Pose pairs and Masks are generated")
            if st.button('Execute'):
                with open("./Database/val_pairs.txt", "w") as f:
                    f.write(user_input + ".jpg " + selected + "_1.jpg")
                predict()
                im = Image.open("./output/second/TOM/val/" + selected + "_1.jpg")
                width, height = im.size
                left = width / 3
                top = 2 * height / 3
                right = 2 * width / 3
                bottom = height
                im1 = im.crop((left, top, right, bottom))
                im1 = im1.resize((256, 256))
                im1 = im1.filter(ImageFilter.SHARPEN)
                im1.save("./output/second/TOM/val/" + selected + "_1.jpg")
                execute_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.08)
                    execute_bar.progress(percent_complete + 1)
                result = Image.open("./output/second/TOM/val/" + selected + "_1.jpg")
                st.image(result, caption="Result", width=200, use_column_width=False)
        else:
            st.error("Please select an item from the library and enter the user name.")
else:
    st.error("Please upload a photo of the person.")

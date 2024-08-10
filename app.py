import streamlit as st
from PIL import Image, ImageFilter
from try_on_clothes.script import predict
import time
from try_on_clothes.utils.upscale import up_scale_x3_normal_fast
from human_parsing.evaluate_human_parsing import execute
from pose_map.pose_parser import pose_parse
from cloth_mask.evaluate_mask import execute_mask
import io

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
            bar = st.progress(0)
            person.save("./Database/val/person/" + user_input + ".jpg")
            cloth = Image.open(uploaded_cloth)
            cloth = cloth.resize((192, 256))
            st.image(cloth, caption="new_cloth_1", width=100, use_column_width=False)
            bar = st.progress(0)
            cloth.save("./Database/val/cloth/" + "new_cloth_1" + ".jpg")
            start = time.time()
            execute_mask()
            progress_bar = st.progress(0)
            pose_parse(user_input)
            execute()
            with open("./Database/val_pairs.txt", "w") as f:
                f.write(user_input + ".jpg " + selected + "_1.jpg")
            predict()
            im = Image.open("./Database/val/tryon-person/" + selected + "_1.jpg")
            im = im.filter(ImageFilter.SHARPEN)
            im.save("./Database/val/tryon-person/" + selected + "_1.jpg")
            result = up_scale_x3_normal_fast("./Database/val/tryon-person/" + selected + "_1.jpg", "./Database/val/tryon-person/" + selected + "_1.jpg")
            result = Image.open("./Database/val/tryon-person/" + selected + "_1.jpg")
            st.image(result, caption="Result", width=200, use_column_width=False)
            st.write(f"The duration of the process is {time.time()- start}")

            st.balloons()
            st.snow()

            img_buffer = io.BytesIO()
            result.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            st.download_button(
                label="Download image",
                data=img_bytes,
                file_name=selected + "_1.jpg",
                mime="image/jpeg"
            )
                
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
            bar = st.progress(0)
            person.save("./Database/val/person/" + user_input + ".jpg")
            start = time.time()
            pose_parse(user_input)
            execute()
            with open("./Database/val_pairs.txt", "w") as f:
                f.write(user_input + ".jpg " + selected + "_1.jpg")
            predict()
            im = Image.open("./Database/val/tryon-person/" + selected + "_1.jpg")
            im = im.filter(ImageFilter.SHARPEN)
            im.save("./Database/val/tryon-person/" + selected + "_1.jpg")
            result = up_scale_x3_normal_fast("./Database/val/tryon-person/" + selected + "_1.jpg", "./Database/val/tryon-person/" + selected + "_1.jpg")
            result = Image.open("./Database/val/tryon-person/" + selected + "_1.jpg")
            st.image(result, caption="Result", width=200, use_column_width=False)
            st.write(f"The duration of the process is {time.time()- start}")

            st.balloons()
            st.snow()

            img_buffer = io.BytesIO()
            result.save(img_buffer, format="JPEG")
            img_bytes = img_buffer.getvalue()

            st.download_button(
                label="Download image",
                data=img_bytes,
                file_name=selected + "_1.jpg",
                mime="image/jpeg"
            )
        else:
            st.error("Please select an item from the library and enter the user name.")
else:
    st.error("Please upload a photo of the person.")

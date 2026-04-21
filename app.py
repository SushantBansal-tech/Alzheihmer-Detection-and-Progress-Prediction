import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import plotly.graph_objects as go
import plotly.express as px


# ================= LOAD MODEL =================

alzheimer_model = tf.keras.models.load_model("alzheimer_heatmap_model.h5")
mri_model = tf.keras.models.load_model("final_mri_detector.keras")

alzheimer_model.build((None,128,128,3))
alzheimer_model(np.zeros((1,128,128,3)))


# ================= CLASSES =================

classes = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]


# ================= REGIONS =================

regions = {
    "NonDemented": "No brain damage",
    "VeryMildDemented": "Hippocampus affected",
    "MildDemented": "Temporal lobe affected",
    "ModerateDemented": "Multiple brain regions affected"
}


# ================= BRAIN REMEDIES =================

brain_remedies = {
    "Hippocampus": {"recovery":70,"risk":40,"tips":["Memory exercise","Doctor check","Brain games","Good sleep"]},
    "Frontal Lobe": {"recovery":60,"risk":50,"tips":["Mental therapy","Routine","Medicine","Family support"]},
    "Left Brain": {"recovery":65,"risk":45,"tips":["Speech therapy","Brain exercise","Doctor","Diet"]},
    "Right Brain": {"recovery":60,"risk":55,"tips":["Therapy","Exercise","Doctor","Monitoring"]},
    "Cerebellum": {"recovery":50,"risk":60,"tips":["Physio","Support","Care","Treatment"]}
}


# ================= METER =================

def show_meter(percent, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "orange"},
                {'range': [70, 100], 'color': "red"},
            ],
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


# ================= GRADCAM FIXED =================

def make_gradcam_heatmap(img_array):

    # 🔥 find last conv layer automatically
    last_conv_layer = None
    for layer in reversed(alzheimer_model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer
            break

    grad_model = tf.keras.Model(
        inputs=alzheimer_model.inputs,
        outputs=[last_conv_layer.output, alzheimer_model.output]
    )

    with tf.GradientTape() as tape:

        conv_outputs, preds = grad_model(img_array)

        # 🔥 FIX: handle functional model output
        if isinstance(preds, list):
            preds = preds[0]

        class_index = int(np.argmax(preds[0]))

        loss = preds[:, class_index]   # ✅ now safe

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = heatmap.numpy()

    heatmap = np.maximum(heatmap, 0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


# ================= REGION % =================

def get_region_percentages(heatmap):

    if len(heatmap.shape) == 3:
        heatmap = heatmap[:,:,0]

    h, w = heatmap.shape

    left = heatmap[:, :w//3]
    right = heatmap[:, 2*w//3:]
    top = heatmap[:h//3, :]
    bottom = heatmap[2*h//3:, :]
    center = heatmap[h//3:2*h//3, w//3:2*w//3]

    total = np.sum(heatmap) + 1e-8

    return {
        "Left Brain": np.sum(left)/total*100,
        "Right Brain": np.sum(right)/total*100,
        "Frontal Lobe": np.sum(top)/total*100,
        "Cerebellum": np.sum(bottom)/total*100,
        "Hippocampus": np.sum(center)/total*100,
    }


# ================= UI =================

st.title("Alzheimer detection and Progression Prediction")

file = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])


if file is not None:

    img = Image.open(file).convert("RGB")

    st.image(img, use_container_width=True)

    if st.button("Predict"):

        # ---------- MRI CHECK ----------
        img_224 = np.array(img.resize((224,224))) / 255.0
        img_224 = np.expand_dims(img_224,0)

        val_pred = mri_model.predict(img_224)[0]

        mri_classes = ["MRI","Non_MRI"]
        val_class = mri_classes[np.argmax(val_pred)]
        val_conf = np.max(val_pred)*100

        st.write(f"MRI Check: {val_class} ({round(val_conf,2)}%)")

        if val_class == "Non_MRI":
            st.error("❌ This is NOT an MRI image")
            st.stop()

        # ---------- ALZHEIMER ----------
        img_128 = np.array(img.resize((128,128)))
        img_np = img_128.copy()

        img_array = img_128 / 255.0
        img_array = np.expand_dims(img_array,0)

        prediction = alzheimer_model.predict(img_array)[0]

        idx = np.argmax(prediction)
        result = classes[idx]
        conf = np.max(prediction)*100

        st.success(result)
        st.write("Confidence:",round(conf,2),"%")
        st.write(regions[result])

        # ---------- HEATMAP ----------
        heatmap = make_gradcam_heatmap(img_array)

        percent_data = get_region_percentages(heatmap)
        part = max(percent_data, key=percent_data.get)

        heatmap = cv2.resize(heatmap,(128,128))
        heatmap = np.uint8(255*heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(img_np,0.6,heatmap,0.4,0)

        st.subheader("Brain Heatmap")
        st.image(overlay, use_container_width=True)

        # ---------- % ----------
        st.subheader("Brain Damage Distribution")

        for k,v in percent_data.items():
            st.write(k,":",round(v,2),"%")

        fig = px.bar(
            x=list(percent_data.keys()),
            y=list(percent_data.values()),
            title="Brain Damage %"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------- REMEDIES ----------
        info = brain_remedies.get(part, {"recovery":50,"risk":50,"tips":["Consult doctor"]})

        show_meter(info["recovery"],"Recovery %")
        show_meter(info["risk"],"Risk %")

        st.subheader("Remedies")

        for t in info["tips"]:
            st.write("✔",t)
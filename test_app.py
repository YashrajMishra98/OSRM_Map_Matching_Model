import streamlit as st

st.title("Hello Streamlit!")
st.write("If you see this, Streamlit is working.")

if st.button("Click Me"):
    st.write("Button clicked!")
    print("Button clicked in terminal!")
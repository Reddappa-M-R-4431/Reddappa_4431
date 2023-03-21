if nav == "Contribute":
    st.header("Contribute to our dataset")
    ex = st.number_input("Enter your age",0.0,100.0)
    sal = st.number_input("Enter your medv",0.00, 50.00,step=1000.0)
    if st.button("Submit"):
        st.success("Submitted")
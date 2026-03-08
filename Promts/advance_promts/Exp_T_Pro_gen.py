from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    template="""

    You are highly skilled financial assistant.an AI Streamlit developer.

    Create a Streamlit application using the following structure:

    - Title: EXPENSE TRACKER TEMPLATE
    - Columns:
    • Date of Payment
    • Method of Payment
    • Paid To
    • Description
    • Amount Paid
    • Running Total

    Application requirements:
    1. Display the title at the top of the page.
    2. Create an input form with fields matching all the columns listed above
    (Date picker, text inputs, number input for amount).
    3. Store submitted entries in memory (session state).
    4. Automatically calculate:
    - Running Total as a cumulative sum of "Amount Paid"
    - Total to Date as the final running total
    5. Display all entries in a table formatted like an expense tracker.
    6. Show "Total to Date" clearly at the top or side of the app.
    7. Use clean and professional Streamlit layout and formatting.
    8. Ensure the app runs using:
    streamlit run prompttest.py

    Output:
    - Generate complete, executable Streamlit Python code
    - Code should be ready to run without modification

    Now generete the final responce.
""",
innput_variables = ["Date Of Payment", "Method Of Payment", "Paid To", "Discription", "Amount Paid", "Running Total"],

)
template.save("exptracker.json")
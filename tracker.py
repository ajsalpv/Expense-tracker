import streamlit as st
import pandas as pd
from datetime import datetime
import pytesseract
from PIL import Image
import speech_recognition as sr
from groq import Groq
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer, Image as ReportImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
import io
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
import urllib.parse

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client using environment variable
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Database connection using SQLAlchemy
@st.cache_resource
def get_db_connection():
    # URL-encode the password to handle special characters like '@'
    encoded_password = urllib.parse.quote(os.getenv("DB_PASSWORD"))
    conn_string = f"postgresql://{os.getenv('DB_USER')}:{encoded_password}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    engine = create_engine(conn_string)
    return engine.connect()

# Groq reasoning function
def groq_reasoning(prompt):
    completion = client.chat.completions.create(
        model="deepseek-r1-distill-llama-70b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
        max_completion_tokens=1024,
        top_p=0.95,
        stream=False,
        reasoning_format="raw"
    )
    return completion.choices[0].message.content

# Parse expense with Groq
def parse_expense(input_text):
    prompt = f"Extract amount, category, and description from: '{input_text}'. Suggest a category if unclear. Reason through your choice."
    result = groq_reasoning(prompt)
    try:
        amount = float(result.split("Amount: $")[1].split(",")[0])
        category = result.split("Category: ")[1].split(",")[0]
        desc = result.split("Description: ")[1]
        return amount, category, desc
    except:
        return 0, "Uncategorized", input_text

# Parse receipt image
def parse_receipt_image(image):
    text = pytesseract.image_to_string(Image.open(image))
    prompt = f"Extract amount, category, and description from this receipt text: '{text}'. Reason through your extraction."
    result = groq_reasoning(prompt)
    try:
        amount = float(result.split("Amount: $")[1].split(",")[0])
        category = result.split("Category: ")[1].split(",")[0]
        desc = result.split("Description: ")[1]
        return amount, category, desc
    except:
        return 0, "Uncategorized", "Receipt"

# Load and save data with SQLAlchemy
def load_data():
    conn = get_db_connection()
    df = pd.read_sql("SELECT * FROM expenses", conn)
    conn.close()
    return df

def save_data(date, amount, category, desc, tags):
    conn = get_db_connection()
    with conn.connection.cursor() as cur:  # Use raw psycopg2 cursor for execute
        cur.execute(
            "INSERT INTO expenses (date, amount, category, description, tags) VALUES (%s, %s, %s, %s, %s)",
            (date, amount, category, desc, tags)
        )
    conn.commit()
    conn.close()

# Generate PDF report with graph
def generate_report(df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("Expense Report", styles['Title']))
    elements.append(Spacer(1, 12))

    data = [["Date", "Amount", "Category", "Description", "Tags"]] + df.values.tolist()
    table = Table(data, colWidths=[80, 60, 80, 150, 100])
    table.setStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ])
    elements.append(table)
    elements.append(Spacer(1, 20))

    df["date"] = pd.to_datetime(df["date"])
    monthly = df.groupby([df["date"].dt.month, "category"])["amount"].sum().unstack().fillna(0)
    monthly.index = [datetime(2025, int(m), 1).strftime("%B") for m in monthly.index]
    plt.figure(figsize=(10, 5))
    monthly.plot(kind="bar", stacked=True)
    plt.title("Monthly Expenses by Category")
    plt.xlabel("Month")
    plt.ylabel("Amount ($)")
    plt.tight_layout()
    graph_buffer = io.BytesIO()
    plt.savefig(graph_buffer, format="png")
    plt.close()
    graph_buffer.seek(0)
    elements.append(ReportImage(graph_buffer, width=500, height=250))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit app
st.title("AI Expense Tracker with Groq Reasoning & PostgreSQL")

st.sidebar.subheader("Budget Settings")
budget = st.sidebar.number_input("Monthly Budget ($)", min_value=0.0, value=500.0)
tags_filter = st.sidebar.multiselect("Filter by Tags", options=["Work", "Personal", "Travel", "Other"])

st.subheader("Add Expense")
expense_input = st.text_input("Tell me about your expense (e.g., 'Spent $20 on coffee')")
if st.button("Add Expense"):
    amount, category, desc = parse_expense(expense_input)
    category = st.selectbox("Confirm Category", [category, "Food/Drink", "Transport", "Entertainment"], key="cat")
    tags = st.multiselect("Add Tags", ["Work", "Personal", "Travel", "Other"], key="tags")
    save_data(datetime.now().strftime("%Y-%m-%d"), amount, category, desc, ",".join(tags))
    st.success("Expense added!")

st.subheader("Upload Audio Expense")
audio_file = st.file_uploader("Upload an audio file (e.g., WAV)", type=["wav"])
if audio_file:
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"You said: {text}")
            amount, category, desc = parse_expense(text)
            save_data(datetime.now().strftime("%Y-%m-%d"), amount, category, desc, "")
            st.success("Expense added!")
        except sr.UnknownValueError:
            st.error("Couldn’t understand the audio.")

st.subheader("Upload Receipt")
receipt = st.file_uploader("Upload a receipt image", type=["jpg", "png"])
if receipt:
    amount, category, desc = parse_receipt_image(receipt)
    save_data(datetime.now().strftime("%Y-%m-%d"), amount, category, desc, "")
    st.success("Receipt added!")

st.subheader("Recurring Expenses")
with st.form("recurring"):
    rec_desc = st.text_input("Description (e.g., Rent)")
    rec_amount = st.number_input("Amount ($)", min_value=0.0)
    rec_day = st.number_input("Day of Month", min_value=1, max_value=31)
    if st.form_submit_button("Add Recurring"):
        today = datetime.now()
        if today.day == rec_day:
            save_data(today.strftime("%Y-%m-%d"), rec_amount, "Recurring", rec_desc, "")
        st.success(f"Recurring expense '{rec_desc}' set for day {rec_day}")

df = load_data()
if not df.empty and tags_filter:
    df = df[df["tags"].str.contains("|".join(tags_filter), na=False)]

if not df.empty:
    total_spent = df["amount"].sum()
    st.write(f"Total Spent: ${total_spent:.2f}")
    if total_spent > budget * 0.9:
        st.warning(f"You’re at {total_spent/budget*100:.1f}% of your ${budget} budget!")
        prompt = f"Reason through my expenses {df.to_dict()} and budget ${budget} to suggest savings tips."
        suggestions = groq_reasoning(prompt)
        st.write("Reasoned Savings Tips:", suggestions)

st.subheader("Monthly Expenses Breakdown")
df["date"] = pd.to_datetime(df["date"])
years = sorted(df["date"].dt.year.unique())
selected_year = st.selectbox("Select Year", options=years, index=len(years)-1)
df_filtered = df[df["date"].dt.year == selected_year]
monthly_by_category = df_filtered.groupby([df_filtered["date"].dt.month, "category"])["amount"].sum().unstack().fillna(0)
monthly_by_category.index = [datetime(selected_year, int(m), 1).strftime("%B") for m in monthly_by_category.index]
st.bar_chart(monthly_by_category, height=400)

if st.button("Get Trends and Insights"):
    prompt = f"Reason through my spending patterns: {df.groupby('category')['amount'].sum().to_dict()} and provide insights."
    trends = groq_reasoning(prompt)
    st.write("Reasoned Insights:", trends)

if st.button("Get Personalized Suggestions"):
    prompt = f"Reason step-by-step through my expenses {df.to_dict()} and suggest tailored savings advice."
    suggestions = groq_reasoning(prompt)
    st.write("Reasoned Suggestions:", suggestions)

st.subheader("Your Expenses")
st.dataframe(df)

st.subheader("Export/Import CSV")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download Expenses as CSV", csv, "expenses.csv", "text/csv")
uploaded_file = st.file_uploader("Import Expenses from CSV", type="csv")
if uploaded_file:
    df_upload = pd.read_csv(uploaded_file)
    conn = get_db_connection()
    with conn.connection.cursor() as cur:
        for _, row in df_upload.iterrows():
            cur.execute(
                "INSERT INTO expenses (date, amount, category, description, tags) VALUES (%s, %s, %s, %s, %s)",
                (row["date"], row["amount"], row["category"], row["description"], row["tags"])
            )
    conn.commit()
    conn.close()
    st.success("Imported successfully!")

st.subheader("Download Expense Report")
if st.button("Generate Report"):
    report_buffer = generate_report(df)
    st.download_button(
        label="Download Report as PDF",
        data=report_buffer,
        file_name=f"expense_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )
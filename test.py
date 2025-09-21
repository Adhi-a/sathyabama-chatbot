from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


app = Flask(__name__)
CORS(app)  # Allow requests from any origin


template = """
You are an Human assistant designed to help students and parents learn about Sathyabama University.Your role is to provide clear, accurate, and friendly information about the college,including admissions, courses, fees, campus life, hostel facilities, placements, and other related topics.Always keep responses concise, easy to understand, and professional, while maintaining a helpful and approachable tone.If a question is unclear or outside your scope, politely guide the user to check the official university website for more details.


User Information:
{user_info}

Here is the conversation history: {context}

Question: {question}

Answer:
"""

user_info = """
Sathyabama Institute of Science and Technology (Sathyabama University) is a private deemed-to-be university located in Chennai, Tamil Nadu, accredited with an A++ grade by NAAC and recognised by UGC, AICTE, and other professional councils.
The university offers a wide range of undergraduate, postgraduate, and doctoral programs including B.E/B.Tech, B.Sc, BBA, MBA, MCA, M.Tech, M.Sc, and professional courses like Pharmacy, Dental Surgery, and Physiotherapy.
Admissions are usually based on merit or entrance exams such as SAEEE for engineering and NEET for certain professional programs, with applications processed online.
The annual fees vary by course, with B.E/B.Tech programs ranging roughly from ₹1.5 to ₹5.5 lakhs per year and MBA programs around ₹4 lakhs in total.
The 140-acre campus on OMR, Chennai, provides modern facilities including separate hostels, laboratories, research centers, a library, and sports infrastructure.
Placements are strong, with top recruiters such as TCS, Infosys, Cognizant, and Wipro, average packages around ₹5.5 LPA, and higher packages for top performers.
Sathyabama consistently ranks among India’s leading institutions in engineering and higher education, making it a reputed choice for students seeking quality academics, infrastructure, and career opportunities.
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

import traceback

@app.route("/chatbot", methods=["POST", "OPTIONS"])
def chatbot():
    if request.method == "OPTIONS":
        return jsonify({"message": "CORS preflight check passed."}), 200

    try:
        data = request.json  # Parse JSON from the request
        print("Received data:", data)  # Debug log

        question = data.get("question", "")
        context = data.get("context", "")
        print("Question:", question)  # Debug log

        result = chain.invoke({"user_info": user_info, "context": context, "question": question})
        print("Response:", result)
        context += f"\nUser: {question}\nAI: {result}"

        return jsonify({"answer": result})  # Ensure the response has the "answer" key
    except Exception as e:
        print("Error occurred:", str(e))  # Log the error for debugging
        print("Full traceback:")
        print(traceback.format_exc())  # This will print the full error traceback
        return jsonify({"error": "Internal server error"}), 500




if __name__ == "__main__":
    app.run(debug=True)

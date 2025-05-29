import PyPDF2
import re
import os
from typing import List, Dict
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import gradio as gr

# Load API key from .env file
load_dotenv("key.env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

class ResumeAnalyzer:
    def __init__(self):
        # Initialize skill and project databases
        self.initialize_skill_databases()

    def initialize_skill_databases(self):
        # Predefined list of technical skills to search for in resumes
        self.technical_skills = [
            "Python", "Java", "JavaScript", "C++", "C#", "SQL", "HTML", "CSS",
            "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas", "NumPy",
            "React", "Angular", "Vue", "Node.js", "Django", "Flask", "FastAPI",
            "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Git",
            "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
            "Data Analysis", "Data Visualization", "Tableau", "Power BI"
        ]

        # Project suggestions mapped to specific skills
        self.project_mappings = {
            "Python": [
                {"level": "Beginner", "idea": "Build a TODO app with Flask/Django"},
                {"level": "Intermediate", "idea": "Create a web scraper using BeautifulSoup"},
                {"level": "Advanced", "idea": "Develop a machine learning model for stock prediction"}
            ],
            "JavaScript": [
                {"level": "Beginner", "idea": "Build a simple calculator web app"},
                {"level": "Intermediate", "idea": "Create a weather app with API integration"},
                {"level": "Advanced", "idea": "Develop a real-time chat application with Socket.io"}
            ],
            "Machine Learning": [
                {"level": "Beginner", "idea": "Implement a linear regression model from scratch"},
                {"level": "Intermediate", "idea": "Build an image classifier with TensorFlow"},
                {"level": "Advanced", "idea": "Create a recommendation system for movies/products"}
            ],
            "Data Analysis": [
                {"level": "Beginner", "idea": "Analyze and visualize COVID-19 data using Pandas"},
                {"level": "Intermediate", "idea": "Perform sentiment analysis on Twitter data"},
                {"level": "Advanced", "idea": "Build a dashboard for real-time data analytics"}
            ]
        }

        # Grouping of skills by domain/category
        self.category_mappings = {
            "Web Development": ["HTML", "CSS", "JavaScript", "React", "Angular", "Vue", "Node.js"],
            "Data Science": ["Python", "Pandas", "NumPy", "Data Analysis", "Data Visualization"],
            "Machine Learning": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn"],
            "Cloud Computing": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"]
        }

    def extract_text_from_pdf(self, file_path: str) -> str:
        # Open and read all text from a PDF file
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])

    def extract_text_from_txt(self, file_path: str) -> str:
        # Open and read all text from a TXT file
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

    def detect_skills(self, text: str) -> List[str]:
        # Normalize text for case-insensitive search
        text_lower = text.lower()
        found_skills = []

        # Search for each technical skill in the text
        for skill in self.technical_skills:
            if re.search(r'\\b' + re.escape(skill.lower()) + r'\\b', text_lower):
                found_skills.append(skill)

        # Add education background if mentioned
        education_keywords = {
            "Bachelor": "Undergraduate",
            "Master": "Graduate",
            "PhD": "PhD",
            "Doctorate": "PhD"
        }

        for keyword, level in education_keywords.items():
            if keyword.lower() in text_lower:
                found_skills.append(f"Education: {level}")

        return found_skills

    def estimate_experience_level(self, text: str) -> str:
        # Use regex to find mentions of years of experience
        matches = re.findall(r"(\\d+)\\s*(years?|yrs?)\\s*(of)?\\s*experience", text.lower())
        if matches:
            years = max(int(match[0]) for match in matches)
            if years >= 5:
                return "Advanced"
            elif years >= 2:
                return "Intermediate"
        return "Beginner"

    def generate_projects(self, skills: List[str], experience_level: str = None) -> List[Dict]:
        projects = []
        used_ideas = set()

        # Add projects based on specific skills and experience level
        for skill in skills:
            if skill in self.project_mappings:
                for project in self.project_mappings[skill]:
                    if experience_level and project["level"] != experience_level:
                        continue
                    key = f"{skill}-{project['level']}-{project['idea']}"
                    if key not in used_ideas:
                        projects.append(project)
                        used_ideas.add(key)

        # Add category-level projects
        for category, category_skills in self.category_mappings.items():
            if any(skill in category_skills for skill in skills):
                category_projects = self.project_mappings.get(category, [])
                for project in category_projects:
                    if experience_level and project["level"] != experience_level:
                        continue
                    key = f"{category}-{project['level']}-{project['idea']}"
                    if key not in used_ideas:
                        projects.append(project)
                        used_ideas.add(key)

        return projects

    def generate_projects_with_gpt(self, skills: List[str], experience_level: str) -> str:
        try:
            # Check if API key is available
            if not GEMINI_API_KEY:
                return "GEMINI_API_KEY not found in environment."

            # Set up Gemini API with key
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.0-flash")

            # Build prompt and get response
            prompt = (
                f"I'm analyzing a resume with these skills: {', '.join(skills)}. "
                f"The estimated experience level is: {experience_level}. "
                f"Suggest 3 creative, realistic project ideas suitable for this profile. "
                "Respond with just the list."
            )

            response = model.generate_content(prompt)
            return response.text

        except Exception as e:
            return f"Failed to get Gemini suggestions: {str(e)}"

    def analyze_resume(self, file_path: str, use_gpt: bool = False) -> Dict:
        # Determine file type and extract text accordingly
        if file_path.endswith(".pdf"):
            text = self.extract_text_from_pdf(file_path)
        elif file_path.endswith(".txt"):
            text = self.extract_text_from_txt(file_path)
        else:
            return {"error": "Unsupported file type"}

        # Detect skills and experience
        skills = self.detect_skills(text)
        experience_level = self.estimate_experience_level(text)

        # Generate project suggestions
        if use_gpt:
            projects = self.generate_projects_with_gpt(skills, experience_level)
        else:
            projects = self.generate_projects(skills, experience_level)

        return {
            "skills": skills,
            "experience_level": experience_level,
            "suggested_projects": projects
        }

# Initialize the analyzer
analyzer = ResumeAnalyzer()

# Define Gradio interface function


def analyze(file, use_gpt):
    try:
        return analyzer.analyze_resume(file, use_gpt)
    except Exception as e:
        return {"error": str(e)}

with gr.Blocks(css="""
    .header-text {font-size: 32px; font-weight: bold; margin-bottom: 10px; color: #1a73e8;}
    .description-text {font-size: 18px; margin-bottom: 20px; color: #555;}
    .result-box {background-color: #f0f4ff; border-radius: 10px; padding: 15px; font-family: monospace; white-space: pre-wrap;}
""") as iface:
    gr.Markdown("<h1 class='header-text'>🚀 AI-Powered Resume Analyzer</h1>")
    gr.Markdown("<p class='description-text'>Upload your resume in PDF or TXT format and get tailored project suggestions based on your skills and experience level.<br><br><i>Enable Gemini for creative AI-generated project ideas.</i></p>")

    with gr.Row():
        file_input = gr.File(label="📄 Upload Resume (.pdf or .txt)", file_types=['.pdf', '.txt'], type="filepath")
        use_gpt = gr.Checkbox(label="✨ Use Gemini for AI suggestions")

    analyze_btn = gr.Button("Analyze Resume", variant="primary")

    output_skills = gr.Textbox(label="🛠️ Detected Skills", interactive=False)
    output_experience = gr.Textbox(label="📈 Estimated Experience Level", interactive=False)
    output_projects = gr.Textbox(label="💡 Suggested Projects", interactive=False, lines=8)

    def process_and_format(file, use_gpt):
        result = analyze(file, use_gpt)
        if "error" in result:
            return "", "", f"❌ Error: {result['error']}"
        skills = ", ".join(result["skills"]) if result["skills"] else "No skills detected."
        experience = result["experience_level"]
        projects = ""
        if isinstance(result["suggested_projects"], str):
            # GPT generated projects
            projects = result["suggested_projects"]
        else:
            for p in result["suggested_projects"]:
                projects += f"- [{p['level']}] {p['idea']}\n"
            if not projects:
                projects = "No project suggestions available."
        return skills, experience, projects

    analyze_btn.click(process_and_format, inputs=[file_input, use_gpt], outputs=[output_skills, output_experience, output_projects])

iface.launch()

import json
from typing import List, Dict
from fastapi import HTTPException
from pydantic import BaseModel, Field
import base64
import re
from github import Github
from github import GithubException
from langchain.chains.llm import LLMChain, PromptTemplate
from langchain_openai import ChatOpenAI

with open('config.json') as f:
    config = json.load(f)

GITHUB_ACCESS_TOKEN = config["GITHUB_ACCESS_TOKEN"]
OPENAI_API_KEY = config["OPENAI_API_KEY"]

if not GITHUB_ACCESS_TOKEN:
    raise ValueError("GITHUB_ACCESS_TOKEN environment variable is not set")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4o-mini")

SUPPORTED_FILE_EXTENSIONS = [
    '.js', '.jsx', '.ts', '.tsx',
    '.html', '.htm',
    '.css', '.scss', '.sass',
    '.java', '.cs',
    '.py', '.sql',
    '.c', '.cpp', '.h', '.hpp',
    '.vb', '.aspx', '.cshtml', '.vbhtml'
]

FORBIDDEN_FOLDERS = [
    'node_modules',     # JavaScript/TypeScript
    '.next',            # Next.js (React framework)
    '__pycache__',      # Python
    'venv', 'env',      # Python virtual environments
    'bin', 'obj',       # C#/.NET build folders
    'build', 'dist',    # Common build/distribution folders
    'target',           # Java/Maven build folder
    'vendor',           # PHP/Composer dependencies
    '.vs', '.vscode',   # Visual Studio/VS Code folders
    'packages',         # NuGet packages folder
    'bower_components', # Bower components (less common now, but still used)
    'jspm_packages',    # JSPM packages
    'tmp', 'temp',      # Temporary folders
    'logs',            # Log folders
    '.sass-cache',      # Sass cache
    '.tsbuildinfo',     # TypeScript build info
    'out',              # Common output folder
    'Debug', 'Release', # C++/C# build configurations
    '.idea',            # JetBrains IDEs folder
    '.gradle',          # Gradle build folder
    'migrations',       # Database migrations folder
]

class AnalysisRequest(BaseModel):
    github_url: str = Field(..., description="The GitHub URL of the repository to analyze")
    requirement: str = Field(..., description="The project requirements")
    question_count: int = Field(..., ge=1, le=20, description="Number of questions to generate (1-20)")

def clean_input(text: str) -> str:
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def should_process_path(path: str) -> bool:
    path_parts = path.lower().split('/')
    if any(folder.lower() in path_parts for folder in FORBIDDEN_FOLDERS):
        return False
    
    return any(path.lower().endswith(ext.lower()) for ext in SUPPORTED_FILE_EXTENSIONS)

def fetch_github_code(github_url: str, token: str) -> str:
    parts = github_url.split('/')
    owner, repo = parts[-2], parts[-1]
    g = Github(token)

    try:
        repository = g.get_repo(f"{owner}/{repo}")
        contents = repository.get_contents("")
        code_content = ""

        while contents:
            file_content = contents.pop(0)
            
            if file_content.type == "dir":
                if not any(folder.lower() in file_content.path.lower() for folder in FORBIDDEN_FOLDERS):
                    contents.extend(repository.get_contents(file_content.path))
                continue

            if should_process_path(file_content.path):
                try:
                    file_data = base64.b64decode(file_content.content).decode('utf-8')
                    code_content += f"// File: {file_content.path}\n{file_data}\n\n"
                except Exception as e:
                    print(f"Warning: Error decoding file {file_content.path}: {str(e)}")
                    continue

        if not code_content:
            raise HTTPException(status_code=404, detail="No supported files found in the repository.")
        return code_content

    except GithubException as e:
        if e.status == 401:
            raise HTTPException(status_code=401, detail="GitHub authentication failed. Please check your access token.")
        elif e.status == 404:
            raise HTTPException(status_code=404, detail="GitHub repository not found. Please check the URL.")
        else:
            raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")

def analyze_code(code: str, requirement: str) -> Dict:
    template = """
    You are an expert code analyzer. Given the following code and requirement, 
    analyze the alignment between them. Provide an alignment score (0-100) and 
    a detailed summary explaining the alignment.

    Code:
    {code}

    Requirement:
    {requirement}

    Return your response as a JSON object with keys 'alignmentScore' and 'alignmentSummary', without any additional formatting or characters.
    """
    prompt = PromptTemplate(template=template, input_variables=["code", "requirement"])
    chain = LLMChain(prompt=prompt, llm=llm)

    response = chain.run({"code": code, "requirement": requirement})
    return json.loads(response)

def generate_questions(code_summary: str, requirement: str, question_count: int) -> List[Dict]:
    template = """
    Based on the following code summary and project requirement, generate {question_count} highly specific technical questions. 
    Each question must reference a particular function, file, or code section and ask how it addresses the given requirement. 
    Also the question should expect short answers which will be answered within 200 seconds.   
    For example, if the requirement mentions implementing two-factor authentication and the code contains a function 
    `implement2FactorAuthentication` in `authentication.ts`, the question should be: 
    'Can you explain how the `implement2FactorAuthentication` function in `authentication.ts` implements two-factor authentication?'

    For each question, provide a 1-2 line description of what the reviewer should check. The questions should focus on how 
    specific code elements solve the given requirements, ensuring key features, functions, or logic are covered.
    
    Code Summary:
    {code_summary}
    
    Requirement:
    {requirement}
    
    Ensure the questions address the specific implementation details mentioned in the requirement and match with the code.
    
    Return your response as a JSON array of objects, each with 'question' and 'lookingFor' keys, without any additional formatting or characters.
    """

    prompt = PromptTemplate(template=template, input_variables=["code_summary", "requirement", "question_count"])
    chain = LLMChain(prompt=prompt, llm=llm)

    response = chain.run({"code_summary": code_summary, "requirement": requirement, "question_count": question_count})
    return json.loads(response)
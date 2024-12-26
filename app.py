from functions import *
from fastapi import FastAPI, HTTPException
import uvicorn

class AnalysisResponse(BaseModel):
    alignment_score: int
    alignment_summary: str
    questions_list: List[Dict[str, str]]

class AnalysisRequest(BaseModel):
    github_url: str = Field(..., description="The GitHub URL of the repository to analyze")
    curriculum: str = Field(..., description="The project requirements")
    question_count: int = Field(..., ge=1, le=20, description="Number of questions to generate (1-20)")

app = FastAPI()

@app.post("/analyse-github", response_model=AnalysisResponse)
async def analyse_github_code(request: AnalysisRequest):
    try:
        github_url = clean_input(request.github_url)
        requirement = clean_input(request.curriculum)
        code_data = fetch_github_code(github_url, GITHUB_ACCESS_TOKEN)
        analysis_result = analyze_code(code_data, requirement)
        questions = generate_questions(analysis_result['alignmentSummary'], requirement, request.question_count)

        response = AnalysisResponse(
            alignment_score=analysis_result['alignmentScore'],
            alignment_summary=analysis_result['alignmentSummary'],
            questions_list=questions
        )

        return response

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    
if __name__ == "__main__":
    uvicorn.run("app:app", port=8080, reload=True)
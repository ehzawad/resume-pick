# Resume Sorting Agent System (RSAS)

AI-powered resume ranking system using OpenAI's GPT-5.1 Response API with multi-agent architecture.

## Overview

RSAS is a sophisticated resume ranking system that processes hundreds of resumes through 8 specialized AI agents to provide accurate, fair, and actionable candidate rankings.

### Key Features

- **8 Specialized Agents**: Job Understanding, Parser, Skills Extraction, Matching, Scoring, Ranking, Bias Check, and Output
- **GPT-5.1 with High Reasoning**: Uses OpenAI's latest reasoning model for maximum accuracy
- **Multi-Dimensional Scoring**: Technical skills, experience, education, culture fit, career trajectory
- **Bias Detection**: Automated fairness analysis and recommendations
- **Full Observability**: Complete trace storage for audit and debugging
- **Idempotent Processing**: Safe resume capability with input hashing
- **Scalable Architecture**: Process 350+ resumes with concurrent batching

## Architecture

```
Job Description → Agent 1 (Job Understanding) → EnrichedJobProfile
                                                       ↓
Resume PDFs → Agent 2 (Parser) → ParsedResume
                                       ↓
              Agent 3 (Skills Extraction) → CandidateProfile
                                                       ↓
                            Agent 4 (Matching) → MatchReport
                                                       ↓
                             Agent 5 (Scoring) → ScoreCard
                                                       ↓
All ScoreCards → Agent 6 (Ranking) → RankedList
                                           ↓
                Agent 7 (Bias Check) → BiasReport
                                           ↓
        Agent 8 (Output) → FormattedOutput (Executive Summary, Reports)
```

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key with access to GPT-5.1

### Setup

1. **Clone the repository**:
   ```bash
   cd /Users/ehz/resume-ranker
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENAI_API_KEY
   ```

5. **Initialize database**:
   ```bash
   python rsas_cli.py init-db
   ```

## Usage

### CLI Commands

#### 1. Process Resumes for a Job

```bash
python rsas_cli.py process \
  --job-id "ml-engineer-2024-01" \
  --description sample_jobs/senior_ml_engineer.txt \
  --resumes resumesets/ \
  --output results/ml_engineer_rankings.json
```

**Parameters**:
- `--job-id` / `-j`: Unique identifier for this job opening
- `--description` / `-d`: Path to job description text file
- `--resumes` / `-r`: Directory containing resume PDFs
- `--output` / `-o`: (Optional) Path to save output JSON

#### 2. Check Pipeline Status

```bash
python rsas_cli.py status --job-id "ml-engineer-2024-01"
```

Shows:
- Current pipeline status (running, completed, failed)
- Current stage being processed
- Progress (resumes processed / total)
- Start time

#### 3. View Rankings

```bash
python rsas_cli.py ranking \
  --job-id "ml-engineer-2024-01" \
  --top-n 20 \
  --export rankings.csv
```

**Parameters**:
- `--job-id` / `-j`: Job identifier
- `--top-n` / `-n`: Number of top candidates to display (default: 10)
- `--export` / `-e`: (Optional) Export rankings to CSV

### Example Workflow

```bash
# 1. Initialize database (first time only)
python rsas_cli.py init-db

# 2. Process resumes for ML Engineer role
python rsas_cli.py process \
  -j "ml-eng-001" \
  -d sample_jobs/senior_ml_engineer.txt \
  -r resumesets/ \
  -o results/ml_rankings.json

# 3. Check status during processing
python rsas_cli.py status -j "ml-eng-001"

# 4. View top 10 candidates
python rsas_cli.py ranking -j "ml-eng-001" -n 10

# 5. Export full rankings
python rsas_cli.py ranking -j "ml-eng-001" -n 100 -e rankings.csv
```

## Configuration

### Configuration Hierarchy

RSAS uses a multi-level configuration system:

1. `config/default.yaml` - Base configuration
2. `config/environments/{env}.yaml` - Environment-specific overrides
3. Environment variables (`RSAS_*` prefix) - Runtime overrides

### Key Configuration Options

```yaml
openai:
  model: "gpt-5.1"
  reasoning_effort: "high"  # low, medium, high
  max_retries: 3
  timeout_seconds: 120

pipeline:
  max_concurrent_resumes: 10  # Process N resumes in parallel
  checkpoint_interval: 10     # Save state every N resumes

agents:
  scoring:
    weights:
      technical_skills: 0.40
      experience: 0.30
      education: 0.15
      culture_fit: 0.10
      career_trajectory: 0.05

database:
  url: "sqlite+aiosqlite:///rsas.db"  # Or PostgreSQL for production
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."

# Optional overrides
export RSAS_OPENAI_MODEL="gpt-5.1"
export RSAS_PIPELINE_MAX_CONCURRENT_RESUMES="5"
export RSAS_DATABASE_URL="postgresql+asyncpg://user:pass@localhost/rsas"
```

## Output Format

### Executive Summary
- Overview of candidate pool quality
- Top tier highlights
- Key trends observed
- Bias report findings

### Top Candidates Summary
For each top candidate:
```json
{
  "name": "Candidate 1",
  "rank": 1,
  "score": 87.5,
  "summary": "Strong ML background with 6 years experience...",
  "key_strengths": ["Deep learning expertise", "Production ML experience"],
  "considerations": ["Limited cloud experience"]
}
```

### Full Report
Complete structured data including:
- All scores and rankings
- Detailed scorecards
- Match reports
- Bias analysis
- Processing metadata

### Recruiter Notes
- Action items for next steps
- Interview recommendations
- Diversity considerations
- Screening suggestions

## Cost Estimation

Based on GPT-5.1 with high reasoning effort:

- **Per Resume**: ~$0.08 - $0.10
- **350 Resumes**: ~$28 - $35
- **Plus Job Understanding**: ~$1 - $2

**Total per job**: ~$29 - $37

## Performance

- **Throughput**: 10 concurrent resumes (configurable)
- **Processing Time**: ~3-5 seconds per resume
- **Total Time (350 resumes)**: ~3-5 minutes

## Project Structure

```
rsas/
├── core/
│   ├── agents/          # 8 specialized agents
│   ├── models/          # Pydantic + SQLAlchemy models
│   ├── orchestrator/    # Pipeline coordination
│   ├── storage/         # Database management
│   └── config/          # Configuration loader
├── integrations/        # OpenAI Response API client
├── observability/       # Structured logging
└── cli/                 # Typer CLI interface

config/                  # YAML configurations
sample_jobs/            # Example job descriptions
resumesets/             # Place resume PDFs here
results/                # Output directory
```

## Agents Deep Dive

### 1. Job Understanding Agent
**Input**: Raw job description text
**Output**: Structured requirements (must-haves, nice-to-haves, experience, education)
**Purpose**: Extract and categorize job requirements for precise matching

### 2. Parser Agent
**Input**: PDF resume file
**Output**: Structured resume sections (contact, education, experience, skills)
**Purpose**: Convert unstructured PDFs to structured data

### 3. Skills Extraction Agent
**Input**: Parsed resume
**Output**: Candidate profile with skills, evidence, years of experience
**Purpose**: Extract skills with supporting evidence and temporal context

### 4. Matching Agent
**Input**: Candidate profile + Job profile
**Output**: Match report (coverage, gaps, alignment scores)
**Purpose**: Compute candidate-job fit metrics

### 5. Scoring Agent
**Input**: Match report + Profiles
**Output**: Multi-dimensional scorecard
**Purpose**: Score candidates on 5 dimensions with weighted average

### 6. Ranking Agent
**Input**: All scorecards
**Output**: Ranked list with tiers (top_10, top_25, top_50, bottom_50)
**Purpose**: Assign ranks, percentiles, and tier classifications

### 7. Bias Check Agent
**Input**: Ranked list
**Output**: Bias report with fairness score and recommendations
**Purpose**: Detect concentration biases, score clustering, tier imbalances

### 8. Output Agent
**Input**: Ranked list + Bias report
**Output**: Formatted reports for different audiences
**Purpose**: Generate executive summaries and actionable insights

## Database Schema

### Key Tables

- **job_profiles**: Structured job requirements
- **candidates**: Candidate profiles and skills
- **scorecards**: Multi-dimensional scores
- **agent_traces**: Full audit trail of all agent executions
- **pipeline_states**: Checkpoint data for resume capability

### Idempotency

All agent executions are cached via unique constraint:
```sql
UNIQUE(job_id, agent_type, input_hash)
```

Rerunning the same job with same inputs uses cached results instantly.

## Troubleshooting

### OpenAI API Errors

**Rate Limits**: Reduce `max_concurrent_resumes` in config
```yaml
pipeline:
  max_concurrent_resumes: 5  # Lower concurrency
```

**Timeouts**: Increase timeout or reduce reasoning effort
```yaml
openai:
  timeout_seconds: 180
  reasoning_effort: "medium"  # Instead of "high"
```

### PDF Parsing Failures

- Ensure PDFs are text-based (not scanned images)
- Check PDF isn't password-protected
- Try alternative parsing with PyPDF2 fallback (automatic)

### Low Quality Results

- Use `reasoning_effort: "high"` for best accuracy
- Ensure job descriptions are detailed and specific
- Check scoring weights align with role priorities

## Development

### Run Tests
```bash
pytest tests/ -v
```

### Type Checking
```bash
mypy rsas/
```

### Code Formatting
```bash
black rsas/
ruff check rsas/ --fix
```

## License

MIT License - see LICENSE file

## Support

For issues and questions:
- GitHub Issues: [repository URL]
- Email: support@example.com

## Changelog

### v1.5 (2024-11-26)
- Initial release with 8-agent architecture
- GPT-5.1 Response API integration
- Full trace storage and idempotency
- CLI interface with Typer
- Bias detection and fairness analysis

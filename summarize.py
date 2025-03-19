import os, logging, time, math, glob, json
from datetime import timedelta
from tqdm import tqdm
from google import genai
from google.genai import types
from google.api_core import retry

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

output_dir = "jfk_json"
schema_json = "schema.json"
model = "gemini-2.0-flash"
max_rpm = 15  # requests per minute

# Function to build GenAI schema object from JSON
def build_schema_from_json(json_data):
    t = json_data.get("type")
    match t:
        case "object":
            properties = {}
            for prop_name, prop_data in json_data["properties"].items():
                properties[prop_name] = build_schema_from_json(prop_data)
            return genai.types.Schema(
                type=genai.types.Type.OBJECT,
                required=json_data.get("required", json_data["required"]),
                properties=properties
            )
        case "string":
            return genai.types.Schema(
                type=genai.types.Type.STRING,
                description=json_data.get("description", "")
            )
        case "array":
            return genai.types.Schema(
                type=genai.types.Type.ARRAY,
                description=json_data.get("description", ""),
                items=build_schema_from_json(json_data["items"])
            )
        case _:
            raise ValueError(f"Unsupported type: {t}")

# Load schema JSON file
schema_file_path = os.path.join(os.path.dirname(__file__), schema_json)
with open(schema_file_path, 'r', encoding='utf-8') as schema_file:
    schema_data = json.load(schema_file)
response_schema = build_schema_from_json(schema_data)

generate_content_config = types.GenerateContentConfig(
    temperature=1,
    top_p=0.95,
    top_k=40,
    max_output_tokens=8192,
    response_mime_type="application/json",
    response_schema=response_schema,
    system_instruction=[
        types.Part.from_text(text="""Please output JSON according to the schema."""),
    ],
)

# Display retry status
logger = logging.getLogger("google.api_core.retry")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

@retry.Retry(initial=10)
def generate_content_retry(text):
    response = client.models.generate_content(
        model=model,
        config=generate_content_config,
        contents=text,
    )
    return response.text

interval = 60 + 1  # with margin
timestamps = []

# Limit the number of requests per minute
def generate_content(text):
    # Wait due to rate limiting
    if 0 < max_rpm <= len(timestamps):
        t = timestamps[-max_rpm]
        if (td := time.monotonic() - t) < interval:
            wait = math.ceil((interval - td) * 10)
            print()
            print(f"Waiting {wait/10} seconds...")
            if (wait_1 := wait % 10):
                time.sleep(wait_1 / 10)
            if wait >= 10:
                for _ in tqdm(range(wait // 10)):
                    time.sleep(1)
    # Get the response
    rtext = generate_content_retry(text)
    timestamps.append(time.monotonic())
    return rtext

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

md_files = sorted(glob.glob('jfk_text/*.md'))
for md_file in tqdm(md_files, desc="Processing files"):
    json_file = os.path.join(output_dir, os.path.basename(md_file).removesuffix('.md') + '.json')
    if os.path.exists(json_file):
        continue
    with open(md_file, 'r', encoding='utf-8') as f:
        text = f.read()
    json_output = generate_content(text)
    json_data = json.loads(json_output)
    with open(json_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(json_data, indent=2, ensure_ascii=False))

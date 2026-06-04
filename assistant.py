import os
import json
import gradio
from dotenv import load_dotenv
from groq import Groq
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
groq_client = Groq()

MODEL_NAME = "openai/gpt-oss-120b"
MAX_TOKEN_LIMIT = 1024
TEMP = 0.3

def respond(message, history):
    extract(message)                    

    for field in questions:             
        if slots[field] is None:
            return questions[field]     

    calculate_price(slots)              
    return "All done, here's your final price."

def build_index(pdf_path):
    
    pages = PyPDFLoader(pdf_path).load()

    
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    ).split_documents(pages)


    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        client.delete_collection("pdf")  
    except Exception:
        pass
    collection = client.get_or_create_collection("pdf")

    collection.add(
        documents=[c.page_content for c in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        metadatas=[{"page": c.metadata.get("page", 0)} for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks.")
    return collection

BASE_VARIANTS = {
    "Utility Wagon 2-Seat":                  {"col_idx": 0, "base_price": 64500, "category": "Wagon"},
    "Utility Wagon 5-Seat":                  {"col_idx": 1, "base_price": 65000, "category": "Wagon"},
    "Utility Wagon Trialmaster (5-Seat)":    {"col_idx": 2, "base_price": 72500, "category": "Wagon"},
    "Utility Wagon Fieldmaster (5-Seat)":    {"col_idx": 3, "base_price": 72500, "category": "Wagon"},
    "Station Wagon Trialmaster (5-Seat)":    {"col_idx": 4, "base_price": 76000, "category": "Wagon"},
    "Station Wagon Fieldmaster (5-Seat)":    {"col_idx": 5, "base_price": 76000, "category": "Wagon"},
    "Quartermaster (5-Seat)":                {"col_idx": 6, "base_price": 66215, "category": "Quartermaster"},
    "Quartermaster Trialmaster (5-Seat)":    {"col_idx": 7, "base_price": 73715, "category": "Quartermaster"},
    "Quartermaster Fieldmaster (5-Seat)":    {"col_idx": 8, "base_price": 73715, "category": "Quartermaster"},
}

FACTORY_OPTIONS = {
    "PPP": {"name": "Rough Pack", "prices": [2370, 2370, "Standard", 2370, "Standard", 2370, 2370, "Standard", 2370]},
    "PPZ": {"name": "Smooth Pack", "prices": [1500, 1500, "Standard", "Standard", "Standard", "Standard", 1500, "Standard", "Standard"]},
    "FPM": {"name": "Donny Grey Metallic Paint", "prices": [970, 970, 970, 970, 970, 970, 925, 925, 925]},
    "WBS": {"name": "18\" Alloy Wheels", "prices": [1690, 1690, 1690, "Standard", 1690, "Standard", 1690, 1690, "Standard"]},
    "ISL": {"name": "Leather Trim - Black", "prices": [1835, 1835, 1835, "Standard", 1835, "Standard", 1835, 1835, "Standard"]},
    "VED": {"name": "Integrated Heavy Duty Winch", "prices": [3515, 3515, 3515, 3515, 3515, 3515, 3515, 3515, 3515]}
}

ACCESSORIES = {
    "VDJ": {"name": "Rock Sliders", "Wagon": 883, "Quartermaster": 908},
    "VCI": {"name": "Tailgate Table", "Wagon": 346, "Quartermaster": None}, 
    "VBP": {"name": "Roller Tonneau Cover", "Wagon": None, "Quartermaster": 2268}
}




slots = {
    "customer_intent": None,
    "driving_environment": None,
    "body_style": None,
    "engine_code": None,
    "trim_edition": None,
    "base_price": None,
    "paint_code": None,
    "contrast_options": None,
    "wheel_tyre_codes": None,
    "interior_codes": None,
    "pack_codes": None,
    "hardware_codes": None,
    "accessories_added": None,
    "total_options_value": None,
    "final_otr_price": None
}


questions = {
    "customer_intent": "Will you be using the vehicle primarily for commercial/utility purposes or as a personal passenger vehicle?",
    "driving_environment": "What kind of terrain do you anticipate driving most? (e.g., daily road/highway driving or challenging off-road trails?)",
    "body_style": "Which body style fits your needs? (Utility Wagon 2-Seat/5-Seat, Station Wagon 5-Seat, or Quartermaster Pick-up 5-Seat)",
    "engine_code": "Which BMW 3.0L straight-six engine do you prefer? (Turbo Petrol [GEB] or Twin-Turbo Diesel [GEC])",
    "trim_edition": "Would you prefer the base Standard configuration, the extreme off-road Trialmaster Edition, or the comfort-focused Fieldmaster Edition?",
    "paint_code": "What exterior body colour would you like? We have Solid options (e.g., Scottish White, Magic Mushroom) and Metallic choices (e.g., Sterling Silver, Donny Grey).",
    "contrast_options": "Would you like a contrast painted roof (Scottish White/Inky Black) or a contrast powder-coated ladder frame (HALO Red/Rhino Grey)? You can choose multiple or say 'None'.",
    "wheel_tyre_codes": "Which wheel and tyre setup would you like? Choose a size (17\" or 18\"), style (Steel or Alloy), and tyre type (Standard Bridgestone or BFGoodrich KO2 All-Terrain).",
    "interior_codes": "Let's configure the cabin. What interior trim (Utility or Black/Grey Leather), seat heating, or driver pack upgrades do you want to add?",
    "pack_codes": "Would you like to add any option groups, like the Rough Pack (front/rear diff locks, KO2 tyres) or the Smooth Pack (rear-view camera, park assist, heated mirrors)?",
    "hardware_codes": "Do you want to add built-in utility features? (e.g., Integrated 5.5-Tonne Winch, Towball options, Raised Air Intake, or High Load Auxiliary Switch Panel)",
    "accessories_added": "Are there any bolt-on lifestyle, cargo, or recovery accessories you want to add? Type 'Done' when finished."
}


tools = [
    {
        "type": "function",
        "function": {
            "name": "save_preferences",
            "description": "Save the customer's INEOS configuration preferences and selected options.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_intent": {
                        "type": "string",
                        "enum": ["commercial", "passenger"],
                        "description": "Whether the vehicle is for commercial/utility or personal/passenger/for themselves use."
                    },
                    "driving_environment": {
                        "type": "string",
                        "description": "The primary driving terrain, e.g., highway, heavy off-road."
                    },
                    "body_style": {
                        "type": "string",
                        "enum": [
                            "Utility Wagon 2-Seat", 
                            "Utility Wagon 5-Seat", 
                            "Station Wagon 5-Seat", 
                            "Quartermaster Pick-up 5-Seat"
                        ],
                        "description": "The selected base vehicle body style."
                    },
                    "engine_code": {
                        "type": "string",
                        "enum": ["GEB", "GEC"],
                        "description": "GEB for Turbo Petrol, GEC for Twin-Turbo Diesel."
                    },
                    "trim_edition": {
                        "type": "string",
                        "enum": ["Standard", "Offroad Trialmaster", "Comfort Fieldmaster"],
                        "description": "The chosen trim level."
                    },
                    "paint_code": {
                        "type": "string",
                        "description": "The RRP code for the chosen exterior body colour."
                    },
                    "contrast_options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of RRP codes for contrast roof and ladder frame."
                    },
                    "wheel_tyre_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of RRP codes for chosen wheels, tyres, and locking nuts."
                    },
                    "interior_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of RRP codes for interior trim, seating, and flooring."
                    },
                    "pack_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of RRP codes for added option packs (e.g., Rough Pack, Smooth Pack)."
                    },
                    "hardware_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of RRP codes for built-in utility hardware (winches, towing, power)."
                    },
                    "accessories_added": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "name": {"type": "string"},
                                "price": {"type": "number"}
                            }
                        },
                        "description": "Array of added accessories with their codes, names, and prices."
                    },
                    "base_price": {
                        "type": "number",
                        "description": "The base price of the chosen model and trim in GBP."
                    },
                    "total_options_value": {
                        "type": "number",
                        "description": "The calculated total cost of all added options and accessories in GBP."
                    },
                    "final_otr_price": {
                        "type": "number",
                        "description": "The final on-the-road price in GBP."
                    }
                }
            }
        }
    }
]


def extract(user_message):
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content":
                    "Call save_preferences with the user's car preferences. "
                    "CRITICAL RULE: If the user says 'no', 'none', 'skip', or 'nothing', YOU MUST RECORD IT. "
                    "If they skip an array question (like accessories or packs), return an empty array []. "
                    "If they skip a string question, return the word 'None'. "
                    "Do NOT leave the key out if they explicitly decline, otherwise the system will ask them again forever."
                },
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto", 
            temperature=0.0,
        )

        msg = resp.choices[0].message

        if msg.tool_calls:
            boxes = json.loads(msg.tool_calls[0].function.arguments)
            for key, value in boxes.items():
                if key in slots and value:
                    slots[key] = value
    except Exception as e:
        print("Clarify?")

def calculate_price(filled_slots):

    base = 60000 
    body = filled_slots.get("body_style", "")
    trim = filled_slots.get("trim_edition", "")
    
    for variant_name, data in BASE_VARIANTS.items():
        if body and trim:
            if body.split(" ")[0] in variant_name and trim in variant_name:
                base = data["base_price"]
                break

    
    options_total = 0
    if filled_slots.get("accessories_added"):
        for item in filled_slots["accessories_added"]:
            options_total += item.get("price", 0)


    filled_slots["base_price"] = base
    filled_slots["total_options_value"] = options_total
    filled_slots["final_otr_price"] = base + options_total

    
    print("\n" + "="*50)
    print("FINAL INEOS CONFIGURATION RECEIPT")
    print("="*50)
    print(f"Body Style & Trim:  {body} ({trim})")
    print(f"Base Vehicle Price: £{base:,.2f}")
    print(f"Added Accessories:  £{options_total:,.2f}")
    print("-" * 50)
    print(f"FINAL TOTAL PRICE:  £{filled_slots['final_otr_price']:,.2f}")
    print("="*50 + "\n")
    
    return filled_slots


if __name__ == "__main__":
    collection = build_index(os.getenv("FILE_PATH"))

    print("Welcome to Ineos, let's find you a car.")
    gradio.ChatInterface(respond).launch()      


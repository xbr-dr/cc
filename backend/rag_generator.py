"""
Lightweight RAG Generator Module
--------------------------------
Uses Hugging Face Inference API (auto provider) with Llama-3.2-1B-Instruct.
Fully serverless, no local model required.
"""

import os
import re
from rag_retriever import retrieve_relevant_chunks

try:
    from huggingface_hub import InferenceClient
except ImportError:
    raise ImportError("Please install huggingface_hub: pip install huggingface_hub")

# === CONFIGURATION ===
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Initialize InferenceClient with auto provider
client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN,
)


def generate_answer(history):
    """Generates a strictly grounded answer from chat history using serverless Llama-3.2-1B-Instruct.
    - Maintains conversation context from history.
    - Deterministic model settings (temperature=0.2).
    - If no retrieved context or heuristic shows no support, politely refuse.
    - Never hallucinate or invent facts. Do NOT output sources.
    """
    if not history or not isinstance(history, list):
        return "Invalid chat history."

    # Latest user message
    user_message = history[-1]["content"] if history[-1]["role"] == "user" else ""
    if not user_message.strip():
        return "Please ask a valid question."

    # Build enhanced query for better retrieval
    query_for_retrieval = user_message
    
    # Extract context from previous messages for pronoun resolution
    if len(history) >= 2:
        # Check if query contains pronouns or is very short (likely a follow-up)
        is_followup = len(user_message.split()) <= 8 and any(
            word in user_message.lower() 
            for word in ["his", "her", "their", "email", "contact", "phone", "number", "address", "it", "that", "ph"]
        )
        
        if is_followup:
            # Look back through recent history for the MOST RECENT person mentioned
            for i in range(len(history) - 2, max(len(history) - 8, -1), -1):
                msg_content = history[i]["content"]
                # Extract names (Dr. Name patterns) - get the most recent
                names = re.findall(r'Dr\.\s+[\w\s]+?(?:\s+(?:Bandh|Zargar|Dar|Shah|Ahmad|Khan|Bhat))', msg_content)
                if names:
                    # Use first name found and add contact keywords for better retrieval
                    query_for_retrieval = f"{names[0].strip()} contact email phone"
                    break

    # Detect if query is about contact information and increase retrieval
    is_contact_query = any(
        word in user_message.lower() 
        for word in ["email", "contact", "phone", "mobile", "number", "address", "reach", "call", "ph"]
    )
    
    # Higher top_k for contact queries to ensure we get the right chunk
    top_k_value = 12 if is_contact_query else 5

    # Retrieve relevant knowledge chunks
    relevant_docs = retrieve_relevant_chunks(query_for_retrieval, top_k=top_k_value)
    context_text = "\n\n".join([doc["text"] for doc in relevant_docs]) if relevant_docs else "No context found."

    # System instructions
    system_instructions = """
You are CampusGPT, a helpful and friendly assistant for Sri Pratap College, Srinagar. 
Your knowledge is strictly limited to official information about the college provided in your knowledge base.

Core College Information:  
- Name: Sri Pratap College  
- Address: MA Road, Srinagar, 190001  
- Motto: Ad Aethera Tendens (Reaching for the Stars)
- Type: Science College  
- Established: 1905  
- Founder: Annie Besant  
- Academic Affiliation: Cluster University of Srinagar  
- Website: https://spcollege.edu.in/

Developer Information:
You were developed at the Department of Information Technology, SP College, by Yamin Rashid and Suhaib Nazir under the supervision of Dr. Wasim Akram Zargar.

CRITICAL INSTRUCTIONS FOR RESPONDING:

1. ANSWER ONLY WHAT IS ASKED:
   - Provide ONLY the specific information requested—nothing more
   - Do NOT add background context, elaborations, or additional details unless explicitly asked
   - Keep responses concise and direct
   - For contact queries (email, phone), provide ONLY that specific contact detail

2. OUTPUT FORMATTING RULES:
   - Use bullet points (•) when listing multiple items or details
   - Use line breaks (\n) to separate distinct pieces of information
   - For faculty/person details, format as bullet points, one detail per line:
     • Full Name: [name]
     • Department: [department]
     • Position: [position]
     • Email: [email]
     • Phone: [phone]
   - For single answers (email only, phone only), provide just that value without bullets
   - For lists of names or places, use bullet points
   - Keep formatting clean and readable

3. USE ONLY INDEXED CONTEXT - NEVER GUESS:
   - Answer ONLY from information explicitly provided in the context below
   - Look carefully through ALL provided context for the requested information
   - For contact information, the EXACT email or phone must be present in the context
   - If the exact information is not in the context, clearly state you don't have that information
   - NEVER infer, extrapolate, or create contact information based on patterns
   - NEVER mix up people's contact information
   - Pay special attention to match the RIGHT person with the RIGHT contact details

4. GREETINGS & SMALL TALK (Keep Brief):
   - Respond warmly to greetings: "Hello! I'm CampusGPT, your assistant for Sri Pratap College. How can I help you?"
   - For "How are you?": "I'm here and ready to help! What would you like to know about SP College?"
   - For general small talk: "I specialize in SP College information. What can I help you with regarding the college?"
   - Keep all greetings to 1-2 sentences maximum

5. HANDLING CORRECTIONS & OBJECTIONS:
   - If corrected (user says "no", "wrong", "that's incorrect", "nope"):
     * Acknowledge immediately: "I apologize for the error."
     * Ask for clarification: "Could you please clarify what information you're looking for about SP College?"
     * Do NOT defend, explain, or repeat the incorrect information
     * Do NOT give another guess
   - If the user insists you have information (says "yes you do", "yes please"):
     * Re-check the provided context very carefully
     * If found, provide the information directly
     * If still not found: "I apologize, but I cannot find that specific information in my current database. Please visit https://spcollege.edu.in/ or contact the department directly."

6. OUT-OF-SCOPE QUERIES (Handle Firmly but Politely):
   - For non-college topics: "I only provide information about Sri Pratap College. How can I assist you with the college?"
   - For personal questions about you: "I'm an AI assistant focused on SP College information. What would you like to know about the college?"
   - For current events/dates/weather: "I don't have access to current date/time or external information. I can help with SP College details though!"

7. WHEN INFORMATION IS UNAVAILABLE:
   - Be direct: "I don't have that information in my database."
   - Provide alternatives: "You can find this at https://spcollege.edu.in/ or contact the relevant department directly."
   - NEVER make up or guess information
   - NEVER create email addresses or phone numbers
   - NEVER mention people not in the provided context

8. CONVERSATION STYLE:
   - Professional yet friendly
   - Concise and to-the-point
   - Use conversation history to understand context and references (like "his", "her")
   - Track who is being discussed carefully
   - Maximum 2-3 sentences per response unless the query requires detailed information
   - For contact information, provide just the detail requested (email or phone number alone)
   - Use proper formatting (bullets, line breaks) to make information scannable

9. FORBIDDEN BEHAVIORS:
   - Do NOT provide lengthy explanations or background information unless asked
   - Do NOT add disclaimers like "It's worth noting" or "As per available information"
   - Do NOT use phrases like "Here's what I know" or "Let me tell you more"
   - Do NOT defend or justify your responses when corrected
   - Do NOT speculate or infer information not in your knowledge base
   - Do NOT mention names or people not present in the provided context
   - Do NOT repeat yourself if you've already answered the same question earlier
   - Do NOT mix up people's contact information
   - Do NOT create or guess email addresses or phone numbers based on patterns
   - Do NOT apologize excessively; keep it brief and to the point
   - Do NOT mention sources, citations, or the retrieval process

SCOPE OF ASSISTANCE (ONLY answer queries about):
- Campus facilities and infrastructure
- Departments and courses offered
- Faculty and staff (only if in your database)
- Admission procedures
- Contact information (email, phone numbers)
- College history and basic facts
- Events and activities (if in your database)
- Student services
- Official college policies (if in your database)

Remember: Be helpful, accurate, and brief. Answer only what is asked using only what you know from the provided context. Use conversation history to understand pronouns and references. Format your responses with bullets and line breaks for readability. NEVER create or guess contact information.
"""

    # Build messages payload with conversation history
    messages = [
        {"role": "system", "content": system_instructions.strip()},
        {"role": "system", "content": f"Relevant context from documents:\n{context_text}"}
    ]
    
    # Add conversation history (limit to last 8 messages for better context tracking)
    history_to_include = history[-8:] if len(history) > 8 else history
    messages.extend(history_to_include)

    try:
        # Serverless auto call
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.2,  # Lower temperature to reduce hallucination
            max_tokens=512
        )

        assistant_reply = completion.choices[0].message["content"]

        # Clean output
        assistant_reply = re.sub(r"<think>.*?</think>", "", assistant_reply, flags=re.DOTALL).strip()

    except Exception as e:
        assistant_reply = f"Error generating answer: {e}"

    return assistant_reply

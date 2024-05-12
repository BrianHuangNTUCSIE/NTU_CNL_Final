import transformers
import torch
import requests
from bs4 import BeautifulSoup
import html
import time

# Initialize the model and pipeline
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "torch_dtype": torch.float16,
        "quantization_config": {"load_in_4bit": True},
        "low_cpu_mem_usage": True,
    },
)

# Configuration
FORUM_URL = 'http://ws5.csie.ntu.edu.tw:4567'
API_KEY = 'ffc038a6-a4b7-4631-a7d9-6a06854e4c7c'
CATEGORY_ID = '2'  # Change this to the category you want to listen to
CHECK_INTERVAL = 60  # Time between checks in seconds

def generate_reply(title, discussion_content,tid):
    """
    Generates a reply for a forum thread using the text generation pipeline, including a system prompt to define the chatbot's persona or role, following the structured approach of system and user messages.
    
    Args:
        title (str): The title of the forum thread.
        discussion_content (list): A list of dictionaries, each containing a post's content, username, and user group.
    """
    # Define the chatbot's persona or role with a system message
    system_message = {"role": "system", "content": f"You are an assistant chatbot in a forum. Your name is llama. The users will mention you with \"@llama\". You should pay closer attention to the responses posted by experts in the \"Experts\" group. Don't mention anyone in your reply. The discussion title is {title}"}
    
    # Convert forum thread discussion content into user messages
    user_messages = []
    user_messages.extend([{"role": "user", "content": f"{post['username']} (in group {post['user_group']}): {post['content']}"} for post in discussion_content])
    
    # Combine system and user messages to form the full conversation context
    messages = [system_message] + user_messages
    
    # Generate a prompt for the model based on the structured messages
    prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        truncation_strategy='only_first'
    )

    # Generate a reply using the pipeline
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Extract and print the generated reply
    generated_reply = outputs[0]["generated_text"][len(prompt):]
    print("Generated Reply:", generated_reply)

    return generated_reply

def post_reply_to_forum(tid, reply_content):
    """
    Posts the generated reply to the specified thread on the NodeBB forum.
    
    Args:
        tid (str): The thread ID where the reply will be posted.
        reply_content (str): The content of the reply to be posted.
    """
    url = f"{FORUM_URL}/api/v3/topics/{tid}"
    # Prepare headers with your API key
    headers = {"Authorization": f"Bearer {API_KEY}"}
    data = {"content": reply_content}
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
        print("Reply posted successfully.")
    except requests.RequestException as e:
        print(f"Failed to post reply: {e}")

def fetch_thread_content(tid):
    """
    Fetches the content of a specific discussion thread along with the usernames, their groups,
    and checks if '@llama' is mentioned in any of the posts.
    
    Returns:
        A tuple containing a boolean indicating if '@llama' is mentioned,
        and a list of dictionaries with each dictionary representing a post's content,
        the username, and the user's group.
    """
    headers = {'Authorization': f'Bearer {API_KEY}'}
    url = f'{FORUM_URL}/api/topic/{tid}'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
        data = response.json()

        llama_mentioned = False
        discussion_content = []
        
        for post in data.get('posts', []):
            soup = BeautifulSoup(post['content'], 'html.parser')  # Use BeautifulSoup to parse HTML content
            content = ' '.join(p.get_text() for p in soup.find_all('p'))  # Extract text from each <p> tag and join them with spaces
            username = post.get('user', {}).get('username', 'Unknown')  # Extract the username
            user_group = post.get('user', {}).get('groupTitle', 'Unknown Group')  # Extract the user's group
            
            # Append the post's content, username, and user group to the discussion_content list
            discussion_content.append({
                'content': content,
                'username': username,
                'user_group': user_group
            })

            # Check if '@llama' is mentioned in the post's content
            if '@llama' in content:
                llama_mentioned = True
            
            # If the post is made by 'llama', reset the mention flag
            if username == 'llama':
                llama_mentioned = False
        
        return llama_mentioned, discussion_content
            
    except requests.RequestException as e:
        print(f"Failed to fetch thread content: {e}")
        return None, []

def check_threads():
    """Fetches and lists discussion threads from a specified NodeBB category, including their content, usernames, and groups."""
    headers = {'Authorization': f'Bearer {API_KEY}'}
    url = f'{FORUM_URL}/api/category/{CATEGORY_ID}'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raises an exception for 4XX or 5XX errors
        data = response.json()
        for thread in data.get('topics', []):
            # Unescape HTML entities in the title
            title = html.unescape(thread['title'])
            llama_mentioned, discussion_content = fetch_thread_content(thread['tid'])
            if llama_mentioned:
                # If 'llama' is mentioned and 'llama' has not replied yet, generate a reply
                reply = generate_reply(title, discussion_content, thread['tid'])
                post_reply_to_forum(thread['tid'],reply)

    except requests.RequestException as e:
        print(f"Failed to fetch discussion threads: {e}")

def main():
    """Main function to periodically check the forum."""
    while True:
        print("-"*100)
        print(f"\nChecking forum at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        check_threads()
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Command Line Interface for RAG Chat System
Allows you to interact with the server from terminal.

Usage:
    # Simple one-shot query
    python chat_cli.py "Πώς κάνω restart το PostgreSQL;"
    
    # Interactive mode
    python chat_cli.py --interactive
    
    # With specific chat ID
    python chat_cli.py --chat-id abc-123 "Next question"
    
    # Create new chat session
    python chat_cli.py --new-chat "PostgreSQL Help"
"""
import requests
import argparse
import sys
import json
from typing import Optional

BASE_URL = "http://localhost:8000"

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text: str):
    """Print colored header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}{text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}Error: {text}{Colors.END}", file=sys.stderr)


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}{text}{Colors.END}")


def print_answer(text: str):
    """Print AI answer."""
    print(f"\n{Colors.CYAN}Assistant:{Colors.END}")
    print(text)


def print_sources(sources: list):
    """Print source documents."""
    if not sources:
        return
    
    print(f"\n{Colors.YELLOW}Sources:{Colors.END}")
    for i, src in enumerate(sources, 1):
        score = src.get('relevance_score', 0)
        source_name = src.get('source', 'Unknown')
        print(f"  {i}. {source_name} (relevance: {score:.2f})")


def check_server() -> bool:
    """Check if server is running."""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except Exception:
        return False


def create_chat(title: str = "CLI Chat") -> Optional[str]:
    """Create a new chat session."""
    try:
        response = requests.post(
            f"{BASE_URL}/chats",
            json={"title": title},
            timeout=10
        )
        
        if response.status_code == 201:
            chat_id = response.text.strip('"')
            print_success(f"Created new chat: {chat_id}")
            return chat_id
        else:
            print_error(f"Failed to create chat: {response.status_code}")
            return None
    
    except Exception as e:
        print_error(f"Failed to create chat: {e}")
        return None


def send_message(message: str, chat_id: Optional[str] = None) -> dict:
    """
    Send a message to the server.
    
    Args:
        message: User message
        chat_id: Optional chat ID (if None, uses stateless endpoint)
    
    Returns:
        Response dictionary with answer and sources
    """
    try:
        if chat_id:
            # Use stateful chat endpoint
            url = f"{BASE_URL}/chats/{chat_id}/message"
        else:
            # Use stateless endpoint
            url = f"{BASE_URL}/chat"
        
        print("message: ", message)
        response = requests.post(
            url,
            json={"role": "user", "content": message},
            #timeout=60000  # LLM can take time
        )
        
        if response.status_code == 200:
            print('response:', response)
            return response.json()
        else:
            print_error(f"Server returned error: {response.status_code}")
            print_error(response.text)
            return None
    
    except requests.exceptions.Timeout:
        print_error("Request timed out. Server may be processing or overloaded.")
        return None
    
    except Exception as e:
        print_error(f"Failed to send message: {e}")
        return None


def list_chats():
    """List all available chats."""
    try:
        response = requests.get(f"{BASE_URL}/chats", timeout=10)
        
        if response.status_code == 200:
            chats = response.json()
            
            if not chats:
                print_warning("No chats found.")
                return
            
            print_header("Available Chats:")
            for chat in chats:
                print(f"  ID: {chat['id']}")
                print(f"  Title: {chat['title']}")
                print(f"  Messages: {chat['message_count']}")
                print(f"  Updated: {chat['last_updated']}")
                print()
        else:
            print_error(f"Failed to list chats: {response.status_code}")
    
    except Exception as e:
        print_error(f"Failed to list chats: {e}")


def interactive_mode(chat_id: Optional[str] = None):
    """
    Start interactive chat session.
    
    Args:
        chat_id: Optional existing chat ID
    """
    print_header("🤖 RAG Chat - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session")
    print("Type 'new' to start a new chat")
    print("Type 'list' to see all chats")
    print("Type 'clear' to clear screen")
    print("-" * 50)
    
    current_chat_id = chat_id
    
    while True:
        try:
            # Get user input
            if current_chat_id:
                prompt = f"{Colors.GREEN}You [{current_chat_id[:8]}]:{Colors.END} "
            else:
                prompt = f"{Colors.GREEN}You [stateless]:{Colors.END} "
            
            user_input = input(prompt).strip()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit', 'q']:
                print_success("Goodbye!")
                break
            
            elif user_input.lower() == 'new':
                title = input("Enter chat title (or press Enter for default): ").strip()
                current_chat_id = create_chat(title or "CLI Chat")
                continue
            
            elif user_input.lower() == 'list':
                list_chats()
                continue
            
            elif user_input.lower() == 'clear':
                import os
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            elif user_input.lower() == 'help':
                print("\nCommands:")
                print("  exit/quit - Exit the program")
                print("  new - Create new chat")
                print("  list - List all chats")
                print("  clear - Clear screen")
                print("  help - Show this help")
                continue
            
            elif not user_input:
                continue
            
            # Send message
            print(f"{Colors.YELLOW}⏳ Thinking...{Colors.END}")
            
            result = send_message(user_input, current_chat_id)
            
            print('result:', result)
            if result:
                print_answer(result.get('answer', 'No answer received'))
                print_sources(result.get('sources', []))
            
        except KeyboardInterrupt:
            print("\n")
            print_success("Goodbye!")
            break
        
        except EOFError:
            break


def one_shot_query(message: str, chat_id: Optional[str] = None, show_sources: bool = True):
    """
    Send a single query and display the result.
    
    Args:
        message: User message
        chat_id: Optional chat ID
        show_sources: Whether to show source documents
    """
    print_header("Sending query to server...")
    print(f"{Colors.GREEN}You:{Colors.END} {message}")
    print(f"{Colors.YELLOW}⏳ Waiting for response...{Colors.END}")
    
    result = send_message(message, chat_id)
    
    if result:
        print_answer(result.get('answer', 'No answer received'))
        
        if show_sources:
            print_sources(result.get('sources', []))
        
        # Show metadata if available
        metadata = result.get('metadata', {})
        if metadata:
            print(f"\n{Colors.YELLOW}Metadata:{Colors.END}")
            for key, value in metadata.items():
                print(f"  {key}: {value}")


def main():
    global BASE_URL
    parser = argparse.ArgumentParser(
        description='Command line client for RAG Chat System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick question
  python chat_cli.py "Πώς κάνω restart το PostgreSQL;"
  
  # Interactive mode
  python chat_cli.py -i
  
  # Use specific chat
  python chat_cli.py --chat-id abc-123 -i
  
  # Create new chat and ask question
  python chat_cli.py --new-chat "DB Help" "How do I backup?"
  
  # List all chats
  python chat_cli.py --list
        """
    )
    
    parser.add_argument(
        'message',
        nargs='?',
        help='Message to send (if not in interactive mode)'
    )
    
    parser.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--chat-id',
        type=str,
        help='Use specific chat ID'
    )
    
    parser.add_argument(
        '--new-chat',
        type=str,
        metavar='TITLE',
        help='Create a new chat with given title'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available chats'
    )
    
    parser.add_argument(
        '--no-sources',
        action='store_true',
        help='Hide source documents in output'
    )
    
    parser.add_argument(
        '--url',
        type=str,
        default=BASE_URL,
        help=f'Server URL (default: {BASE_URL})'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output raw JSON response'
    )
    
    args = parser.parse_args()
    
    # Update base URL if provided
    #global BASE_URL
    BASE_URL = args.url.rstrip('/')
    
    # Check server
    if not check_server():
        print_error(f"Cannot connect to server at {BASE_URL}")
        print_error("Make sure the server is running:")
        print("  python setup_and_run.py --run")
        sys.exit(1)
    
    # Handle list command
    if args.list:
        list_chats()
        sys.exit(0)
    
    # Handle new chat creation
    chat_id = args.chat_id
    if args.new_chat:
        chat_id = create_chat(args.new_chat)
        if not chat_id:
            sys.exit(1)
    
    # Interactive mode
    if args.interactive:
        interactive_mode(chat_id)
    
    # One-shot query
    elif args.message:
        result = send_message(args.message, chat_id)
        
        if args.json:
            # Output raw JSON
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Formatted output
            if result:
                print_answer(result.get('answer', 'No answer received'))
                
                if not args.no_sources:
                    print_sources(result.get('sources', []))
            else:
                sys.exit(1)
    
    else:
        # No message and not interactive - show help
        parser.print_help()


if __name__ == "__main__":
    main()
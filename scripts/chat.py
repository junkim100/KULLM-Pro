import os
import time
import torch
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"

    # Standard colors
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"

    # Think token colors (distinctive)
    THINK_START = "\033[38;5;208m"  # Orange
    THINK_CONTENT = "\033[38;5;245m"  # Gray
    THINK_END = "\033[38;5;208m"  # Orange

    # Role colors
    USER = "\033[93m"  # Yellow
    ASSISTANT = "\033[95m"  # Magenta
    SYSTEM = "\033[92m"  # Green


def format_think_tokens(text):
    """Format text with colored think tokens."""
    if not text:
        return text

    # Pattern to match think tokens and their content
    think_pattern = r"(<think>)(.*?)(</think>)"

    def replace_think(match):
        start_tag = match.group(1)
        content = match.group(2)
        end_tag = match.group(3)

        return (
            f"{Colors.THINK_START}{start_tag}{Colors.RESET}"
            f"{Colors.THINK_CONTENT}{content}{Colors.RESET}"
            f"{Colors.THINK_END}{end_tag}{Colors.RESET}"
        )

    # Replace think tokens with colored versions
    formatted_text = re.sub(think_pattern, replace_think, text, flags=re.DOTALL)

    return formatted_text


def load_system_prompt():
    """Load system prompt optimized for think tokens."""
    # Always use think-token optimized prompt for chat interface
    return """You are KULLM-Pro, a helpful assistant developed by Korea University NLP&AI Lab.

You naturally code-switch between English and Korean when reasoning through problems. Show your step-by-step thinking process and provide clear, helpful answers.

Be concise and avoid unnecessary repetition."""


def load_model_and_tokenizer(path):
    """Load model and tokenizer for streaming generation."""
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print("Model and tokenizer loaded")
    return model, tokenizer


def generate_with_streaming(model, tokenizer, input_text, max_new_tokens=512, **kwargs):
    """Generate text with simple streaming output."""
    from transformers import TextIteratorStreamer
    from threading import Thread

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Setup streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    # Generation parameters with repetition penalty
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": kwargs.get("do_sample", True),  # Enable sampling by default
        "temperature": kwargs.get(
            "temperature", 0.7
        ),  # Lower temperature for more focused responses
        "top_p": kwargs.get("top_p", 0.9),
        "repetition_penalty": kwargs.get(
            "repetition_penalty", 1.1
        ),  # Prevent repetition
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": [tokenizer.eos_token_id, 151645],  # Handle multiple EOS tokens
        "streamer": streamer,
    }

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        yield new_text, generated_text

    thread.join()
    return generated_text


def main(
    model_path: str,
    sys_prompt: str = None,
    max_new_tokens: int = 512,
    think_mode: bool = True,
    **kwargs,
):
    print(f"{Colors.CYAN}🚀 Loading KULLM-Pro Chat Interface...{Colors.RESET}")
    print(f"{Colors.BLUE}Model path: {model_path}{Colors.RESET}")

    # Load model and tokenizer with UTF-8 support for Korean
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load system prompt from file if not provided
    if sys_prompt is None:
        sys_prompt = load_system_prompt()
        print(
            f"{Colors.GREEN}📝 Loaded system prompt with think token instructions{Colors.RESET}"
        )

    print(
        f"{Colors.BLUE}BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id}){Colors.RESET}"
    )
    print(
        f"{Colors.BLUE}EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id}){Colors.RESET}"
    )
    print(
        f"{Colors.BLUE}Think mode: {'🧠 Enabled' if think_mode else '❌ Disabled'}{Colors.RESET}"
    )
    messages = []
    messages.append({"role": "system", "content": sys_prompt})

    print(f"\n{Colors.CYAN}💬 KULLM-Pro Chat Interface Ready!{Colors.RESET}")
    print(
        f"{Colors.YELLOW}Commands: 'clear' to reset conversation, 'exit' to quit{Colors.RESET}"
    )
    while 1:
        try:
            # Use colored prompt for input
            input_ = input(f"{Colors.BLUE}👤 Enter instruction: {Colors.RESET}")

            if input_ == "clear":
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                os.system("clear")
                print(f"{Colors.GREEN}🧹 Conversation cleared!{Colors.RESET}\n")
                continue
            elif input_ == "exit":
                print(f"{Colors.CYAN}👋 Goodbye!{Colors.RESET}")
                break
            elif input_.strip() == "":
                continue

            messages.append({"role": "user", "content": input_})
            os.system("clear")

            # Display conversation history with colors
            for m in messages[:-1]:
                role = m["role"]
                content = m["content"]

                if role == "system":
                    print(
                        f"{Colors.SYSTEM}🔧 System: {content[:100]}{'...' if len(content) > 100 else ''}{Colors.RESET}"
                    )
                elif role == "user":
                    print(f"{Colors.USER}👤 User: {content}{Colors.RESET}")
                elif role == "assistant":
                    formatted_content = format_think_tokens(content)
                    print(
                        f"{Colors.ASSISTANT}🤖 Assistant: {formatted_content}{Colors.RESET}"
                    )

            # Show current user input
            print(f"{Colors.USER}👤 User: {input_}{Colors.RESET}")

            start = time.time()

            # Apply chat template with think mode support
            template_kwargs = {"add_generation_prompt": True}
            if think_mode is not None:
                template_kwargs["think_mode"] = think_mode

            text = tokenizer.apply_chat_template(
                messages, tokenize=False, **template_kwargs
            )

            # Generate response with streaming
            print(f"{Colors.ASSISTANT}🤖 Assistant: {Colors.RESET}", end="", flush=True)

            result = ""
            for new_token, _ in generate_with_streaming(
                model, tokenizer, text, max_new_tokens=max_new_tokens, **kwargs
            ):
                result += new_token
                # Simple streaming display - just print each token as it comes
                print(new_token, end="", flush=True)

            print()  # New line after generation

            # Clean up the result - remove unwanted tokens
            result = result.replace("<tool_call>", "").replace("</tool_call>", "")
            result = result.replace("<|im_end|>", "").strip()

            # Since your model doesn't use <think> tags but generates reasoning directly,
            # we'll just display the clean result without trying to add think token colors
            print(f"\033[A\033[K{Colors.ASSISTANT}🤖 Assistant: {result}{Colors.RESET}")

            messages.append({"role": "assistant", "content": result})

            # Show timing info
            elapsed = time.time() - start
            print(f"{Colors.CYAN}⏱️  Response time: {elapsed:.2f}s{Colors.RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}⚠️  Interrupted by user{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}❌ Error: {str(e)}{Colors.RESET}")
            continue


if __name__ == "__main__":
    import fire

    fire.Fire(main)

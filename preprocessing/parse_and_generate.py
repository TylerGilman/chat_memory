import pandas as pd
import anthropic
import json
from datetime import datetime
import time
import os
from typing import List, Dict
from dotenv import load_dotenv
from pathlib import Path


def setup_environment():
    """Setup and verify environment variables with detailed debugging."""
    script_dir = Path(__file__).parent.absolute()
    env_path = script_dir / ".env"

    if env_path.exists():
        print(".env file found!")
        load_dotenv(env_path)
    else:
        print("Warning: .env file not found!")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("API key successfully loaded from .env")
        return api_key
    else:
        print("API key not found in .env file")
        raise ValueError("Please ensure ANTHROPIC_API_KEY is set in your .env file")


class ConversationProcessor:
    def __init__(self, api_key: str, output_dir: str = "transformed_conversations"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.progress_file = self.output_dir / "progress.json"
        self.processed_ids = self._load_progress()

        self.prompt_template = """As an AI assistant summarizing your chat history, I am speaking directly to you (Person1 in the original conversation). Generate today's date (between 2024-01-01 and 2024-12-31) and help me summarize your important conversation history with {other_person}.

    Input Conversation:
    {dialogue}

    Context:
    This conversation was about {topic}. 
    Original summary: {summary}

    Task:
    1. First, generate a current date (today) between 2024-01-01 and 2024-12-31 when this summary is being created.
    2. Then, generate 3-4 conversations that occurred before this date, spanning a realistic period (anywhere from 1 week to 6 months prior to today's date).
    3. Transform the original face-to-face dialogue into the first message exchange
    4. Create follow-up conversations that show how the situation developed
    5. End with a personal summary to "you" (Person1) that:
       - Mentions today's date when the summary is being made
       - References how long ago each conversation happened
       - Highlights the progression of events from your perspective
       - Explains the significance of key developments
       - Uses time markers like "last week", "three months ago", "earlier this spring"

    Format your response as:
    [TODAY'S DATE]
    YYYY-MM-DD

    [COMPREHENSIVE SUMMARY FOR YOU]
    Starting with "Here's a summary of your interactions with [Person2]..." provide a detailed paragraph describing the sequence of events from your perspective, including clear temporal references.

    [CONVERSATIONS]
    [Date: YYYY-MM-DD]
    [HH:MM] You: message
    [HH:MM] {other_person}: message

    [Date: YYYY-MM-DD]
    [HH:MM] {other_person}: message
    [HH:MM] You: message
    (continue for all conversations)

    For example: If today is 2024-05-15, you might reference a first conversation from "three months ago in February", a follow-up "six weeks ago in late March", and the most recent exchange "just last week in early May."
    """

    def _load_progress(self) -> set:
        """Load set of already processed conversation IDs."""
        if self.progress_file.exists():
            with open(self.progress_file, "r") as f:
                progress = json.load(f)
                return set(progress.get("processed_ids", []))
        return set()

    def _save_progress(self, conversation_id: str):
        """Save progress after processing each conversation."""
        self.processed_ids.add(conversation_id)
        progress = {
            "processed_ids": list(self.processed_ids),
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    def _parse_claude_response(self, response_text: str) -> Dict:
        """Parse Claude's response into structured format with today's date and personal summary."""
        # Split into main sections
        sections = response_text.split("[CONVERSATIONS]")
        if len(sections) != 2:
            raise ValueError("Unexpected response format")

        # Parse header section
        header_sections = sections[0].split("[COMPREHENSIVE SUMMARY FOR YOU]")
        if len(header_sections) != 2:
            raise ValueError("Missing summary section")

        # Get today's date
        today_date = header_sections[0].replace("[TODAY'S DATE]\n", "").strip()

        # Get personal summary
        personal_summary = header_sections[1].strip()

        # Parse conversations
        conversations_text = sections[1].strip()
        conversations = []
        current_date = None
        current_messages = []

        for line in conversations_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith("[Date:"):
                if current_date and current_messages:
                    conversations.append(
                        {"date": current_date, "messages": current_messages}
                    )
                current_date = line.replace("[Date:", "").replace("]", "").strip()
                current_messages = []
            elif "[" in line and "]" in line:  # Message line
                timestamp = line[1 : line.index("]")]
                message = line[line.index("]") + 1 :].strip()
                current_messages.append(
                    {"timestamp": f"{current_date} {timestamp}", "content": message}
                )

        # Add last conversation
        if current_date and current_messages:
            conversations.append({"date": current_date, "messages": current_messages})

        return {
            "today_date": today_date,
            "personal_summary": personal_summary,
            "conversations": conversations,
        }

    def _convert_to_finetuning_format(self, parsed_response: Dict) -> str:
        """Convert parsed response to finetuning format with exact formatting."""
        # Get all relevant dates
        dates = [conv["date"] for conv in parsed_response["conversations"]]
        start_date = min(dates)
        end_date = max(dates)

        # Combine all messages in chronological order
        all_messages = []
        for conv in parsed_response["conversations"]:
            for msg in conv["messages"]:
                all_messages.append(msg)

        # Sort messages by timestamp
        all_messages.sort(key=lambda x: x["timestamp"])

        # Format messages as per specification
        formatted_messages = "\n".join(
            [f"{msg['timestamp']} | {msg['content']}" for msg in all_messages]
        )

        # Create the complete formatted string
        formatted_output = f"""[START DATE]
    {start_date}
    [END DATE]
    {end_date}
    [CHAT MESSAGES]
    {formatted_messages}
    [SUMMARY]
    {parsed_response['personal_summary']}"""

        return formatted_output

    def _save_result(self, result: Dict, conversation_id: str, success: bool):
        """Save both original and finetuning format results."""
        status_dir = self.output_dir / ("success" if success else "errors")
        status_dir.mkdir(exist_ok=True)

        # Save original format
        original_file = status_dir / f"conversation_{conversation_id}_original.json"
        with open(original_file, "w") as f:
            json.dump(result, f, indent=2)

        if success:
            # Save finetuning format
            finetuning_file = (
                status_dir / f"conversation_{conversation_id}_finetune.txt"
            )
            finetuning_format = self._convert_to_finetuning_format(
                result["transformed_conversations"]
            )
            with open(finetuning_file, "w") as f:
                f.write(finetuning_format)

    def process_single_conversation(self, row: pd.Series) -> Dict:
        """Process a single conversation through Claude API."""
        conversation_id = str(row["id"])

        if conversation_id in self.processed_ids:
            print(f"Skipping already processed conversation {conversation_id}")
            return None

        # Extract Person2's name from the dialogue if possible
        dialogue_lines = row["dialogue"].split("\n")
        other_person = "the other person"  # default
        for line in dialogue_lines:
            if "#Person2#:" in line:
                # Try to find a name in the first line of Person2's dialogue
                content = line.split("#Person2#:", 1)[1].strip()
                words = content.split()
                for i, word in enumerate(words):
                    if word.startswith(("Mr.", "Mrs.", "Dr.", "Ms.")):
                        other_person = f"{word} {words[i+1]}"
                        break
                break

        prompt = self.prompt_template.format(
            dialogue=row["dialogue"],
            topic=row["topic"],
            summary=row["summary"],
            other_person=other_person,
        )

        try:
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=4096,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )

            parsed_response = self._parse_claude_response(response.content[0].text)

            result = {
                "original_id": conversation_id,
                "topic": row["topic"],
                "original_summary": row["summary"],
                "transformed_conversations": parsed_response,
                "timestamp": datetime.now().isoformat(),
            }

            self._save_result(result, conversation_id, success=True)
            self._save_progress(conversation_id)
            print(f"Successfully processed and saved conversation {conversation_id}")

            return result

        except Exception as e:
            error_result = {
                "original_id": conversation_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "original_data": {
                    "topic": row["topic"],
                    "summary": row["summary"],
                    "dialogue": row["dialogue"],
                },
            }
            self._save_result(error_result, conversation_id, success=False)
            self._save_progress(conversation_id)
            print(f"Error processing conversation {conversation_id}: {str(e)}")
            return None

    def process_all_conversations(
        self, df: pd.DataFrame, batch_size: int = 5
    ) -> List[Dict]:
        """Process all conversations with rate limiting and batching."""
        all_results = []

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i : i + batch_size]
            batch_results = []

            for _, row in batch.iterrows():
                result = self.process_single_conversation(row)
                if result:
                    batch_results.append(result)
                time.sleep(2)

            all_results.extend(batch_results)

        return all_results


def main():
    try:
        api_key = setup_environment()

        input_file = os.getenv("INPUT_CSV", "your_input.csv")
        print(f"Using input file: {input_file}")

        num_rows = os.getenv("NUM_ROWS_TO_PROCESS", "2")
        num_rows = int(num_rows) if num_rows != "all" else None
        print(f"Will process {num_rows if num_rows else 'all'} rows")

        output_dir = os.getenv("OUTPUT_DIR", "transformed_conversations")
        processor = ConversationProcessor(api_key, output_dir=output_dir)

        df = pd.read_csv(input_file, nrows=num_rows)
        print(f"Successfully loaded {len(df)} conversations from CSV")

        results = processor.process_all_conversations(df, batch_size=5)

        if results:
            final_output = Path(output_dir) / "all_results.json"
            with open(final_output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Processing complete. Results saved to {final_output}")
        else:
            print("Processing complete. No successful results to combine.")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    main()

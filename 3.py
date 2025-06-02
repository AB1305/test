import os
import re
import time
import openai
import pandas as pd
from io import StringIO
from tqdm import tqdm
from dotenv import load_dotenv
import gspread
from gspread_dataframe import set_with_dataframe
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials as GoogleCredentials
from flask import Flask, render_template, request, jsonify, redirect, url_for
from threading import Thread
import uuid
from datetime import datetime
import json

# Load environment and API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# In-memory storage for job status (in production, use Redis or database)
job_status = {}

# Constants
CHUNK_SIZE = 50
SHEETS_BATCH_SIZE = 100  # Max rows per Google Sheets batch write
MAX_SHEET_ROWS = 1000000  # Google Sheets row limit
OUTPUT_HEADERS = [
    "Company Name", "Company Size", "Market Cap", "Recent Funding",
    "Areas of Interest", "Relevance to Elucidata's Services",
    "Buying Intent from Elucidata", "Priority Order"
]

PROMPT_TEMPLATE = """You are an analyst tasked with classifying biotechnology companies.

Objective:
Analyze each company to determine its relevance and buying intent related to Elucidata's services, and assign an overall priority for targeting.

Details to Research and Fill:

Company Name: Full official name.
Company Size: Estimated number of employees.
Market Cap: Current market capitalization or "N/A" if private.
Recent Funding: Latest funding round details with amount and year or "N/A".
Areas of Interest: Main therapeutic areas, research domains, or business focus.
Relevance to Elucidata's Services: Rate as High, Medium, or Low based on alignment with Elucidata's data-driven solutions.
Buying Intent from Elucidata: Estimate likelihood of engagement (High, Medium, Low) based on company initiatives.
Priority Order: Assign:

P1 for high relevance and buying intent (top priority).
P2 for moderate relevance or buying intent (follow-up candidates).
P3 for low relevance and low intent (deprioritized).

Note: Ensure data accuracy by cross-referencing company websites, Crunchbase, and other reliable data sources. If information is unavailable, indicate as "N/A".

Input:
{companies}

Output Format:
Create a table with exactly these columns:
Company Name | Company Size | Market Cap | Recent Funding | Areas of Interest | Relevance to Elucidata's Services | Buying Intent from Elucidata | Priority Order

Example:
Example Company | 100-200 | $500M | $50M (2023) | Gene therapy, AI-driven drug discovery | High | High | P1
"""

# Google Sheets setup
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
gc = gspread.authorize(credentials)

# Google Drive API setup for permissions
drive_creds = GoogleCredentials.from_service_account_file("service_account.json", scopes=scope)
drive_service = build('drive', 'v3', credentials=drive_creds)

def extract_sheet_id(url):
    match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    if match:
        return match.group(1)
    raise ValueError("Invalid Google Sheet URL")

def read_sheet(sheet_id):
    try:
        spreadsheet = gc.open_by_key(sheet_id)
        worksheet = spreadsheet.get_worksheet(0)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Remove rows where all values are empty strings
        df = df[~(df.astype(str).eq('').all(axis=1))]
        
        return df
    except Exception as e:
        raise Exception(f"Failed to read Google Sheet: {str(e)}")

def create_output_sheet(title):
    try:
        spreadsheet = gc.create(title)
        set_public_permissions(spreadsheet.id)
        return spreadsheet.get_worksheet(0), spreadsheet.id
    except Exception as e:
        raise Exception(f"Failed to create output sheet: {str(e)}")

def set_public_permissions(sheet_id):
    try:
        drive_service.permissions().create(
            fileId=sheet_id,
            body={"role": "reader", "type": "anyone"},
            fields="id"
        ).execute()
    except Exception as e:
        print(f"Warning: Could not set public permissions: {str(e)}")

def safe_sheet_write(worksheet, data_rows, start_row, max_retries=3):
    """
    Safely write data to Google Sheets with batch size limits and retry logic
    """
    if not data_rows:
        return True, start_row
    
    try:
        # Split data into safe batches
        total_rows = len(data_rows)
        current_row = start_row
        
        for i in range(0, total_rows, SHEETS_BATCH_SIZE):
            batch = data_rows[i:i + SHEETS_BATCH_SIZE]
            batch_size = len(batch)
            
            # Calculate range for this batch
            end_row = current_row + batch_size - 1
            range_name = f'A{current_row}:H{end_row}'
            
            # Try to write this batch with retries
            for attempt in range(max_retries):
                try:
                    worksheet.update(range_name, batch, value_input_option="RAW")
                    print(f"Successfully wrote batch {i//SHEETS_BATCH_SIZE + 1} (rows {current_row}-{end_row})")
                    break
                except Exception as e:
                    if "exceeds grid limits" in str(e).lower() or "max" in str(e).lower():
                        print(f"Hit Google Sheets limit at row {current_row}: {str(e)}")
                        return False, current_row  # Return where we stopped
                    
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 2  # Exponential backoff
                        print(f"Batch write attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to write batch after {max_retries} attempts: {str(e)}")
                        return False, current_row
            
            current_row = end_row + 1
            
            # Small delay between batches to avoid rate limiting
            time.sleep(0.5)
        
        return True, current_row
        
    except Exception as e:
        print(f"Unexpected error in safe_sheet_write: {str(e)}")
        return False, start_row

def classify_chunk(chunk_df):
    try:
        companies = chunk_df.to_csv(index=False)
        prompt = PROMPT_TEMPLATE.format(companies=companies)

        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that classifies company data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        table_str = response.choices[0].message.content.strip()

        # Try to parse the response as a pipe-separated table first
        try:
            # Split by lines and process each line
            lines = table_str.split('\n')
            # Find lines that look like table rows (contain multiple |)
            table_lines = [line for line in lines if line.count('|') >= 3]
            
            if not table_lines:
                raise ValueError("No table structure found in response")
            
            # Join the table lines back
            clean_table = '\n'.join(table_lines)
            df = pd.read_csv(StringIO(clean_table), sep="|", engine="python", skipinitialspace=True)
            
        except Exception as parse_error:
            # Fallback to comma-separated parsing
            try:
                df = pd.read_csv(StringIO(table_str), sep=",", engine="python", skipinitialspace=True)
            except Exception as csv_error:
                return pd.DataFrame(), f"Failed to parse response as table: {str(parse_error)}, {str(csv_error)}"

        # Clean column names
        df.columns = [col.strip() for col in df.columns]
        
        # Remove any completely empty rows
        df = df.dropna(how='all')
        
        # Filter to only expected columns that exist
        available_cols = [col for col in OUTPUT_HEADERS if col in df.columns]
        df = df[available_cols]

        if len(available_cols) < len(OUTPUT_HEADERS):
            missing_cols = set(OUTPUT_HEADERS) - set(available_cols)
            return pd.DataFrame(), f"Missing required columns: {missing_cols}"

        return df, None

    except Exception as e:
        return pd.DataFrame(), f"API/Processing error: {str(e)}"

def create_failed_companies_sheet(job_id):
    """Create a Google Sheet for logging failed companies"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sheet_title = f"Failed_Companies_{job_id[:8]}_{timestamp}"
        
        spreadsheet = gc.create(sheet_title)
        worksheet = spreadsheet.get_worksheet(0)
        
        # Set up headers for failed companies sheet
        headers = ["Company_Name", "Original_Row_Data", "Failure_Reason", "Chunk_Number", "Row_Index", "Timestamp"]
        worksheet.update('A1:F1', [headers])
        
        # Make it publicly readable
        set_public_permissions(spreadsheet.id)
        
        return worksheet, spreadsheet.id
    except Exception as e:
        print(f"Error creating failed companies sheet: {str(e)}")
        return None, None

def detect_company_column(df):
    """Detect the most likely company name column with better accuracy"""
    if df.empty:
        return None
        
    company_keywords = [
        'company', 'name', 'organization', 'firm', 'business', 
        'corp', 'inc', 'ltd', 'entity', 'client', 'account'
    ]
    
    # First priority: exact matches with common company column names
    exact_matches = ['company name', 'company_name', 'companyname', 'name', 'company']
    for col in df.columns:
        col_clean = col.lower().strip().replace(' ', '_')
        if col_clean in exact_matches:
            return col
    
    # Second priority: partial matches in column names
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in company_keywords):
            return col
    
    # Third priority: analyze content of string columns
    string_columns = [col for col in df.columns if df[col].dtype == 'object']
    
    for col in string_columns:
        if len(df[col].dropna()) == 0:
            continue
            
        # Sample first 10 non-null values
        sample_values = df[col].dropna().head(10).astype(str)
        
        # Check for company indicators in the values
        company_indicators = [
            'inc', 'corp', 'ltd', 'llc', 'co.', 'company', 'pharma', 
            'bio', 'tech', 'therapeutics', 'pharmaceutical', 'biotech',
            'medical', 'health', 'lab', 'laboratories', 'research'
        ]
        
        matches = 0
        for val in sample_values:
            val_lower = val.lower()
            if any(indicator in val_lower for indicator in company_indicators):
                matches += 1
        
        # If more than 30% of sampled values contain company indicators
        if matches / len(sample_values) > 0.3:
            return col
    
    # Fourth priority: look for columns with varied, non-numeric content (likely names)
    for col in string_columns:
        if len(df[col].dropna()) == 0:
            continue
            
        unique_values = df[col].nunique()
        total_values = len(df[col].dropna())
        
        # High uniqueness ratio suggests names rather than categories
        if total_values > 0 and (unique_values / total_values) > 0.8:
            return col
    
    # Final fallback: first string column, or first column overall
    if string_columns:
        return string_columns[0]
    elif len(df.columns) > 0:
        return df.columns[0]
    else:
        return None

def log_failed_companies_to_sheet(failed_worksheet, chunk_df, chunk_number, error_reason, missing_count=None, successful_result=None):
    """Log individual company failures to Google Sheet with better data capture"""
    
    if failed_worksheet is None:
        print("Warning: Failed worksheet is None, cannot log failures")
        return 0
    
    try:
        # Detect the company name column
        company_col = detect_company_column(chunk_df)
        
        # Prepare rows to add to the sheet
        failed_rows = []
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if missing_count and successful_result is not None:
            # For partial failures, we need to identify which specific companies failed
            # Get successfully processed company names
            successful_companies = set()
            if 'Company Name' in successful_result.columns:
                successful_companies = set(successful_result['Company Name'].astype(str).str.strip().str.lower())
            
            # Find companies that were in input but not in successful output
            for idx, row in chunk_df.iterrows():
                # Get company name from input
                if company_col and company_col in chunk_df.columns:
                    input_company_name = str(row[company_col]).strip() if pd.notna(row[company_col]) else f"Row_{idx}"
                else:
                    input_company_name = f"Row_{idx}"
                
                # Clean company name
                if input_company_name.strip() == '' or input_company_name.lower() in ['nan', 'none']:
                    input_company_name = f"Unknown_Company_Row_{idx}"
                
                # Check if this company is NOT in successful results
                input_name_clean = input_company_name.lower().strip()
                if input_name_clean not in successful_companies:
                    # This company failed - add to failed list
                    
                    # Get original row data as a string representation
                    row_data_parts = []
                    for col, val in row.items():
                        if pd.notna(val) and str(val).strip():
                            row_data_parts.append(f"{col}: {str(val)}")
                    
                    row_data = " | ".join(row_data_parts) if row_data_parts else "No valid data found"
                    
                    failed_rows.append([
                        input_company_name,
                        row_data,
                        f"PARTIAL_CHUNK_FAILURE: {error_reason}",
                        chunk_number,
                        idx,
                        timestamp
                    ])
        else:
            # For complete failures, log each company
            for idx, row in chunk_df.iterrows():
                # Get company name
                if company_col and company_col in chunk_df.columns:
                    company_name = str(row[company_col]) if pd.notna(row[company_col]) else f"Row_{idx}"
                else:
                    company_name = f"Row_{idx}"
                
                # Clean company name
                if company_name.strip() == '' or company_name.lower() in ['nan', 'none']:
                    company_name = f"Unknown_Company_Row_{idx}"
                
                # Get original row data as a string representation
                row_data_parts = []
                for col, val in row.items():
                    if pd.notna(val) and str(val).strip():
                        row_data_parts.append(f"{col}: {str(val)}")
                
                row_data = " | ".join(row_data_parts) if row_data_parts else "No valid data found"
                
                failed_rows.append([
                    company_name,
                    row_data,
                    f"COMPLETE_CHUNK_FAILURE: {error_reason}",
                    chunk_number,
                    idx,
                    timestamp
                ])
        
        # Add all failed companies to the sheet using safe write
        if failed_rows:
            # Get current row count to append properly
            current_data = failed_worksheet.get_all_values()
            start_row = len(current_data) + 1
            
            # Use safe write function
            success, _ = safe_sheet_write(failed_worksheet, failed_rows, start_row)
            
            if success:
                print(f"Successfully logged {len(failed_rows)} individual failed companies to sheet for chunk {chunk_number}")
            else:
                print(f"Partial logging of failed entries for chunk {chunk_number}")
        
        return len(failed_rows)
        
    except Exception as e:
        print(f"Error logging failed companies for chunk {chunk_number}: {str(e)}")
        return 0

def process_classification(job_id, sheet_url):
    """Background task to process the classification"""
    try:
        job_status[job_id]['status'] = 'processing'
        job_status[job_id]['message'] = 'Extracting sheet ID...'
        
        sheet_id = extract_sheet_id(sheet_url)
        
        job_status[job_id]['message'] = 'Reading input data...'
        df = read_sheet(sheet_id)
        
        if df.empty:
            raise Exception("Input sheet is empty or contains no valid data")
        
        total_rows = len(df)
        chunks = [df[i:i+CHUNK_SIZE] for i in range(0, len(df), CHUNK_SIZE)]
        total_chunks = len(chunks)
        
        job_status[job_id]['message'] = f'Processing {total_rows} companies in {total_chunks} chunks...'
        job_status[job_id]['total_chunks'] = total_chunks
        job_status[job_id]['processed_chunks'] = 0
        job_status[job_id]['total_input_companies'] = total_rows
        
        # Create output sheet for successful classifications
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_title = f"Classified_Companies_{job_id[:8]}_{timestamp}"
        output_ws, output_id = create_output_sheet(output_title)
        
        # Initialize the output sheet with headers
        output_ws.update('A1:H1', [OUTPUT_HEADERS])
        current_output_row = 2
        sheet_full = False  # Track if we hit sheet limits
        
        # Create failed companies sheet - ALWAYS create it
        failed_ws, failed_sheet_id = create_failed_companies_sheet(job_id)
        if failed_sheet_id:
            job_status[job_id]['failed_sheet_url'] = f"https://docs.google.com/spreadsheets/d/{failed_sheet_id}"
        
        failed_chunks = []
        failed_companies_count = 0
        successful_companies = 0
        processed_companies_total = 0
        
        for i, chunk in enumerate(chunks):
            chunk_number = i + 1
            chunk_size = len(chunk)
            job_status[job_id]['message'] = f'Processing chunk {chunk_number}/{total_chunks} ({chunk_size} companies)...'
            
            success = False
            final_error = None
            result_companies_count = 0
            
            # Try processing the chunk up to 3 times
            for attempt in range(3):
                try:
                    result, error_reason = classify_chunk(chunk)
                    
                    if not result.empty and error_reason is None:
                        result_companies_count = len(result)
                        
                        if result_companies_count > 0 and not sheet_full:
                            # Try to add successful results to output sheet using safe write
                            write_success, new_current_row = safe_sheet_write(
                                output_ws, 
                                result.values.tolist(), 
                                current_output_row
                            )
                            
                            if write_success:
                                current_output_row = new_current_row
                                successful_companies += result_companies_count
                                success = True
                                job_status[job_id]['message'] = f'Chunk {chunk_number}/{total_chunks} completed successfully ({result_companies_count} companies classified)'
                            else:
                                # Sheet is full, mark remaining as failed due to sheet limits
                                sheet_full = True
                                final_error = "Google Sheets row limit exceeded - cannot write more data"
                                job_status[job_id]['message'] = f'Sheet limit reached at chunk {chunk_number}. Remaining chunks will be marked as failed due to sheet limits.'
                        elif sheet_full:
                            # We processed the data but can't write it due to sheet limits
                            final_error = "Google Sheets row limit exceeded - data processed but cannot be written"
                            result_companies_count = 0  # Mark as failed since we can't store the results
                        else:
                            success = True  # Empty result but no error
                        
                        break
                    else:
                        final_error = error_reason or "Unknown processing error"
                        job_status[job_id]['message'] = f'Chunk {chunk_number} attempt {attempt+1}/3 failed: {final_error[:100]}...'
                        
                except Exception as e:
                    final_error = f"Exception during processing: {str(e)}"
                    job_status[job_id]['message'] = f'Chunk {chunk_number} attempt {attempt+1}/3 exception: {str(e)[:100]}...'
                
                if attempt < 2:  # Don't sleep on the last attempt
                    time.sleep(2)  # Brief delay between retries
            
            # Track processed companies
            processed_companies_total += chunk_size
            
            # Replace the existing failure handling code in your process_classification function with this:

            # Handle failures - check if we got fewer companies than input
            if not success:
                # Complete chunk failure
                failed_chunks.append(chunk_number)
                
                # Log ALL companies in this chunk as failed
                if failed_ws:
                    companies_logged = log_failed_companies_to_sheet(failed_ws, chunk, chunk_number, final_error)
                    failed_companies_count += companies_logged
                    print(f"Chunk {chunk_number} completely failed: logged {companies_logged} companies")
                else:
                    failed_companies_count += chunk_size
                    print(f"Chunk {chunk_number} failed but couldn't log to sheet")
                    
            elif result_companies_count < chunk_size:
                # Partial failure - some companies in chunk were lost
                missing_count = chunk_size - result_companies_count
                partial_error = f"Partial processing: {missing_count} companies missing from OpenAI response"
                
                if failed_ws:
                    # Log the specific missing companies (pass the successful result for comparison)
                    companies_logged = log_failed_companies_to_sheet(
                        failed_ws, 
                        chunk, 
                        chunk_number, 
                        partial_error, 
                        missing_count,
                        result  # Pass the successful result to identify which companies failed
                    )
                    failed_companies_count += companies_logged  # This should now equal missing_count
                    print(f"Chunk {chunk_number} partial failure: {missing_count} companies missing, logged {companies_logged} individual companies")
                else:
                    failed_companies_count += missing_count
                    print(f"Chunk {chunk_number} partial failure: {missing_count} companies missing")
            
            # Update status
            job_status[job_id]['processed_chunks'] = chunk_number
            job_status[job_id]['progress'] = (chunk_number / total_chunks) * 100
            job_status[job_id]['successful_companies_count'] = successful_companies
            job_status[job_id]['failed_companies_count'] = failed_companies_count
            
            # If sheet is full, mark all remaining chunks as failed
            if sheet_full and chunk_number < total_chunks:
                remaining_chunks = total_chunks - chunk_number
                remaining_companies = sum(len(chunks[j]) for j in range(chunk_number, total_chunks))
                
                job_status[job_id]['message'] = f'Sheet limit reached. Marking remaining {remaining_companies} companies as failed due to Google Sheets limits.'
                
                # Log remaining chunks as failed due to sheet limits
                for j in range(chunk_number, total_chunks):
                    remaining_chunk = chunks[j]
                    remaining_chunk_number = j + 1
                    failed_chunks.append(remaining_chunk_number)
                    
                    if failed_ws:
                        companies_logged = log_failed_companies_to_sheet(
                            failed_ws, 
                            remaining_chunk, 
                            remaining_chunk_number, 
                            "Google Sheets row limit exceeded - chunk not processed"
                        )
                        failed_companies_count += companies_logged
                    else:
                        failed_companies_count += len(remaining_chunk)
                
                break
            
            # Small delay between chunks to avoid rate limiting
            time.sleep(1)
        
        # Final validation - ensure math adds up
        expected_total = total_rows
        actual_total = successful_companies + failed_companies_count
        
        if actual_total != expected_total:
            discrepancy = expected_total - actual_total
            print(f"WARNING: Math discrepancy detected!")
            print(f"Expected: {expected_total}, Actual: {actual_total}, Missing: {discrepancy}")
            
            # Add discrepancy to failed count
            if discrepancy > 0:
                failed_companies_count += discrepancy
                discrepancy_error = f"Discrepancy detected: {discrepancy} companies unaccounted for"
                
                # Log discrepancy if we have a failed sheet
                if failed_ws:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    current_data = failed_ws.get_all_values()
                    next_row = len(current_data) + 1
                    
                    discrepancy_row = [
                        f"DISCREPANCY_{discrepancy}_COMPANIES",
                        f"Math discrepancy: Expected {expected_total} total, got {actual_total}",
                        discrepancy_error,
                        "N/A",
                        "SYSTEM_CHECK",
                        timestamp
                    ]
                    
                    failed_ws.update(f'A{next_row}:F{next_row}', [discrepancy_row])
        
        # Complete the job
        job_status[job_id]['status'] = 'completed'
        job_status[job_id]['output_url'] = f"https://docs.google.com/spreadsheets/d/{output_id}"
        job_status[job_id]['completed_at'] = datetime.now().isoformat()
        job_status[job_id]['failed_companies_count'] = failed_companies_count
        job_status[job_id]['successful_companies_count'] = successful_companies
        job_status[job_id]['total_processed'] = successful_companies + failed_companies_count
        job_status[job_id]['sheet_full'] = sheet_full
        
        # Final message
        if sheet_full:
            job_status[job_id]['message'] = f'Classification completed with sheet limit reached! {successful_companies}/{total_rows} companies classified successfully, {failed_companies_count} companies failed or could not be written due to Google Sheets limits.'
        elif failed_companies_count > 0:
            job_status[job_id]['message'] = f'Classification complete! {successful_companies}/{total_rows} companies classified successfully, {failed_companies_count} companies failed. Check the failed companies sheet for details.'
        else:
            job_status[job_id]['message'] = f'Classification completed successfully! All {successful_companies} companies processed.'
        
        job_status[job_id]['failed_chunks'] = failed_chunks
        
        # Always include failed URL if sheet was created
        if failed_sheet_id:
            job_status[job_id]['failed_url'] = job_status[job_id].get('failed_sheet_url')
        
        # Print final summary for debugging
        print(f"=== JOB {job_id[:8]} SUMMARY ===")
        print(f"Input companies: {total_rows}")
        print(f"Successfully classified: {successful_companies}")
        print(f"Failed companies: {failed_companies_count}")
        print(f"Total accounted: {successful_companies + failed_companies_count}")
        print(f"Failed chunks: {len(failed_chunks)}")
        print(f"Sheet full: {sheet_full}")
   
    except Exception as e:
        job_status[job_id]['status'] = 'error'
        job_status[job_id]['message'] = f'Fatal error: {str(e)}'
        job_status[job_id]['error'] = str(e)
        print(f"Fatal error in job {job_id}: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def start_classification():
    sheet_url = request.form.get('sheet_url', '').strip()
    
    if not sheet_url:
        return jsonify({'error': 'Please provide a Google Sheet URL'}), 400
    
    try:
        # Validate URL format
        extract_sheet_id(sheet_url)
    except ValueError:
        return jsonify({'error': 'Invalid Google Sheet URL format'}), 400
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_status[job_id] = {
        'status': 'queued',
        'message': 'Job queued for processing...',
        'created_at': datetime.now().isoformat(),
        'sheet_url': sheet_url,
        'progress': 0,
        'processed_chunks': 0,
        'total_chunks': 0,
        'successful_companies_count': 0,
        'failed_companies_count': 0
    }
    
    # Start background processing
    thread = Thread(target=process_classification, args=(job_id, sheet_url))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'Job started successfully',
        'status_url': url_for('get_status', job_id=job_id)
    })

@app.route("/status/<job_id>")
def get_status(job_id):
    job = job_status.get(job_id)
    if not job:
        return jsonify({"status": "error", "message": "Invalid job ID"}), 404

    status_data = {
        "status": job["status"],
        "message": job["message"],
        "processed_chunks": job.get("processed_chunks", 0),
        "total_chunks": job.get("total_chunks", 0),
        "progress": job.get("progress", 0),
        "output_url": job.get("output_url", None),
        "failed_url": job.get('failed_url'),
        "successful_companies_count": job.get("successful_companies_count", 0),
        "failed_companies_count": job.get("failed_companies_count", 0),
        "sheet_full": job.get("sheet_full", False)
    }

    return jsonify(status_data)

@app.route('/jobs')
def list_jobs():
    """List all jobs with their current status"""
    return jsonify({
        'jobs': [
            {
                'job_id': job_id,
                'status': info['status'],
                'message': info['message'],
                'created_at': info['created_at'],
                'progress': info.get('progress', 0),
                'successful_companies_count': info.get('successful_companies_count', 0),
                'failed_companies_count': info.get('failed_companies_count', 0),
                'output_url': info.get('output_url'),
                'failed_url': info.get('failed_url'),
                'has_failed_sheet': 'failed_sheet_url' in info,
                'sheet_full': info.get('sheet_full', False)
            }
            for job_id, info in job_status.items()
        ]
    })

app.template_folder = 'templates'

if __name__ == '__main__':
    print("🚀 Starting Company Classification Flask App...")
    print("📝 Make sure you have:")
    print("   - service_account.json file in the same directory")
    print("   - OPENAI_API_KEY in your .env file")
    print("   - Required Python packages installed")
    print("\n🌐 Access the app at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
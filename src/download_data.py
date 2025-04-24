from Bio import Entrez
import pandas as pd
import time
import pandas as pd
import csv
import numpy as np
import os
from tqdm import tqdm
import re
import aiohttp
import json
import asyncio
import nest_asyncio
from Bio import Entrez
import requests
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError

# Global variables
DIR_PATH = None
Entrez.email = 'your.email@example.com'
nest_asyncio.apply()


def fetch_pubmed_data(query, max_entries=100000):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_entries)
    record = Entrez.read(handle)
    handle.close()
    
    # Get the list of PubMed IDs
    id_list = record["IdList"]
    
    # Create a list to hold the data
    pubmed_data = []
    
    # Process IDs in batches to avoid timeout issues
    batch_size = 100
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        try:
            # Fetch details for the batch of PubMed IDs
            ids = ",".join(batch_ids)
            handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records['PubmedArticle']:
                try:
                    # Extract basic information
                    pmid = article['MedlineCitation']['PMID']
                    title = article['MedlineCitation']['Article']['ArticleTitle']
                    abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', ["No abstract available"])[0]
                    
                    # Extract publication date
                    pub_date = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
                    year = pub_date.get('Year', 'N/A')
                    
                    # Extract authors
                    author_list = article['MedlineCitation']['Article'].get('AuthorList', [])
                    authors = "; ".join([
                        f"{author.get('LastName', '')}, {author.get('ForeName', '')}"
                        for author in author_list
                        if 'LastName' in author
                    ]) if author_list else "No authors listed"
                    
                    pubmed_data.append((str(pmid), title, abstract, year, authors))
                    
                except Exception as e:
                    print(f"Error processing article {pmid}: {str(e)}")
            
            # Add a small delay to avoid overwhelming the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error fetching batch: {str(e)}")
            continue
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(pubmed_data, columns=['PMID', 'Title', 'Abstract', 'Year', 'Authors'])
    return df

async def get_pmc_id(session, pmid):
    """Fetch PMC ID for a given PMID."""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?tool=my_tool&email=my_email@example.com&ids={pmid}&format=json"
    async with session.get(url) as response:
        data = await response.json()
        #data = await response.json(content_type='text/html')
        if 'records' in data and data['records']:
            return pmid, data['records'][0].get('pmcid')
        return pmid, None

async def process_pmid_list(pmid_list):
    """Process a list of PMIDs to get their corresponding PMC IDs."""
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [get_pmc_id(session, pmid) for pmid in pmid_list]
        for future in tqdm(asyncio.as_completed(tasks), total=len(pmid_list)):
            pmid, pmcid = await future
            if pmcid:
                results.append({'pmid': pmid, 'pmcid': pmcid})
    return results

# def save_to_csv(results, filename):
#     filepath = os.path.join(DIR_PATH, filename)
#     with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
#         fieldnames = ['pmid', 'pmcid']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)

async def create_pmc_mapping(pmid_list, organism_name, prefix):
    """Create and save PMC to PMID mapping."""
    # Get the results from process_pmid_list
    results = await process_pmid_list(pmid_list)
    
    # Create DataFrame from results
    df = pd.DataFrame(results)
    
    # Generate filename with prefix and organism name
    filename = f"{prefix}_{organism_name}_pmc_mapping.csv"
    
    # Save DataFrame to CSV without index
    df.to_csv(filename, index=False)
    
    return df

def get_pmc_bioc_xml(pmcid):
    """Fetch BioC XML for a given PMC ID."""
    base_url = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/"
    url = f"{base_url}{pmcid}/unicode"
    try:
        response = requests.get(url)
        response.raise_for_status()
        if response.content:
            try:
                return ET.fromstring(response.content)
            except ParseError as e:
                print(f"Error parsing XML for PMCID {pmcid}: {e}")
                print(f"Response content: {response.content[:200]}...")
        else:
            print(f"Empty response for PMCID {pmcid}")
    except requests.RequestException as e:
        print(f"Error retrieving data for PMCID {pmcid}: {e}")
    return None

def get_pubmed_title(pmid):
    """Fetch title for a given PMID from PubMed."""
    handle = Entrez.efetch(db='pubmed', id=str(pmid), retmode='xml')
    record = Entrez.read(handle)
    handle.close()
    try:
        return record['PubmedArticle'][0]['MedlineCitation']['Article']['ArticleTitle']
    except KeyError:
        return 'Title not found'

def parse_pmc_article(root):
    """Parse PMC article XML and extract relevant sections."""
    sections = {
        "TITLE": "",
        "ABSTRACT": [],
        "INTRO": [],
        "RESULTS": [],
        "DISCUSS": []
    }

    # Extract title
    title_element = root.find(".//article-title")
    if title_element is not None:
        sections["TITLE"] = title_element.text.strip()

    # Extract other sections
    passages = root.findall(".//passage")
    for passage in passages:
        section_type = passage.find("infon[@key='section_type']")
        passage_type = passage.find("infon[@key='type']")
        if section_type is not None and passage_type is not None:
            if section_type.text in sections:
                if section_type.text == "ABSTRACT" and passage_type.text in ["paragraph", "abstract"]:
                    text = passage.find("text")
                    if text is not None:
                        sections[section_type.text].append(text.text.strip())
                elif passage_type.text == "paragraph":
                    text = passage.find("text")
                    if text is not None:
                        sections[section_type.text].append(text.text.strip())

    # Join the paragraphs for each section
    for key in ["ABSTRACT", "INTRO", "RESULTS", "DISCUSS"]:
        sections[key] = ' '.join(sections[key])

    return sections

def populate_dataframe(df):
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        pmcid = row['pmcid']
        if pd.isna(pmcid):
            print(f"Skipping row {index}: PMCID is NaN")
            continue
        xml_root = get_pmc_bioc_xml(pmcid)
        if xml_root is not None:
            article_content = parse_pmc_article(xml_root)
            try:
                #df.at[index, 'title'] = article_content['TITLE']
                df.at[index, 'abstract'] = article_content['ABSTRACT']
                df.at[index, 'intro'] = article_content['INTRO']
                df.at[index, 'results'] = article_content['RESULTS']
                df.at[index, 'discuss'] = article_content['DISCUSS']
            except KeyError as e:
                print(f"Error processing PMCID {pmcid}: Missing key {e}")
                #df.at[index, 'title'] = ''
                df.at[index, 'abstract'] = ''
                df.at[index, 'intro'] = ''
                df.at[index, 'results'] = ''
                df.at[index, 'discuss'] = ''
        else:
            print(f"Failed to retrieve or parse XML for PMCID {pmcid}")
            #df.at[index, 'title'] = ''
            df.at[index, 'abstract'] = ''
            df.at[index, 'intro'] = ''
            df.at[index, 'results'] = ''
            df.at[index, 'discuss'] = ''
    
    return df

def create_pmid_line_dict(df):
    """Create a dictionary of PMIDs and their corresponding lines."""
    text_columns = ['title', 'abstract', 'intro', 'results', 'discuss']
    split_pattern = re.compile(r'(?<=[.!?])\s+')
    reference_pattern = re.compile(r'\s*\[\d+(?:,\d+)*\]')

    def process_row(row):
        combined_text = ' '.join(row[text_columns].fillna(''))
        sentences = split_pattern.split(combined_text)
        return [reference_pattern.sub('', sent.strip()) for sent in sentences]

    return dict(zip(df['pmid'], df.apply(process_row, axis=1)))

def prepare_dataset(query,max_entries = 100,  organism_name = "mouse", pmid_col_name = "PMID", save_file_prefix = "rag_covid_test_10", save_dir = "/Users/suchanda/Desktop/workspace_rwth/rag/data"):
    """Prepare the dataset from the input file."""
    df = fetch_pubmed_data(query, max_entries = max_entries)
    # Process PMIDs
    df_all_pmid = list(set(df[pmid_col_name]))
    print(f"Total number of PMIDs : {len(df_all_pmid)}")

    # # Create PMC mapping
    loop = asyncio.get_event_loop()
    df_withpmc_information = loop.run_until_complete(create_pmc_mapping(pmid_list=df_all_pmid, organism_name=organism_name, prefix=save_file_prefix))
    print(df_withpmc_information)


    # Read PMC mapping
    pmc2pmid_mapping_path = os.path.join(save_dir, f"{save_file_prefix}_{organism_name}_pmc2pmid_mapping.csv")
    df_withpmc_information.to_csv(pmc2pmid_mapping_path)
    pmc2pmid_mapping = pd.read_csv(pmc2pmid_mapping_path)
    print(f"Number of articles in available on PMC: {len(pmc2pmid_mapping)}")

    # Populate dataframe with article content
    df_withpmc_information = populate_dataframe(df_withpmc_information)

    # Clean and process dataframe
    df_withpmc_information.replace('', np.nan, inplace=True)

    # Fetch titles
    for index, row in df_withpmc_information.iterrows():
        pmid = row['pmid']
        title = get_pubmed_title(pmid)
        df_withpmc_information.at[index, 'title'] = title



    #df_withpmc_information.dropna(subset=["title","abstract", "intro", "results", "discuss"], how="all", axis=0, inplace=True)
    # Verify no null values
    #assert df_withpmc_information.notnull().all().all(), "DataFrame contains null values"

    # Save intermediate file
    save_path = os.path.join(save_dir, f"{save_file_prefix}_{organism_name}_withpmc_information.csv")
    df_withpmc_information.to_csv(save_path, index=False)
    print(f"Finished saving the line dataframe with title, abstract, intro, result and discussion ....")

    # Create line dictionary
    print("Creating the line dictionary...")
    pmid_line_dict = create_pmid_line_dict(df_withpmc_information)

    # Verify dictionary
    assert len(pmid_line_dict) == len(set(df_withpmc_information["pmid"].to_list())), "Mismatch in the number of PMIDs in the dataframe and the line dictionary"

    # Save line dictionary
    line_dictionary_save_loc = os.path.join(save_dir, f"{save_file_prefix}_{organism_name}_pmc_line_dict.json")
    with open(line_dictionary_save_loc, 'w') as f:
        json.dump(pmid_line_dict, f)

    print(f"Finished creating the line dictionary..")
    print(f"saved it at {line_dictionary_save_loc}..")






def run(query):
    prepare_dataset(query=query, max_entries=10, organism_name="human", 
                    save_dir="/Users/suchanda/Desktop/workspace_rwth/rag/data",
                    save_file_prefix = "test_kidney_lr_human", 
                    )
    print(f"*****HOOOMAN DONE*************")


    # human_input_path = "/work/suchanda/ligand_receptor_mining/src/celltalkdb/celltalk_human_pubmed.csv"
    # prepare_dataset(human_input_path, organism_name="human")
    # print(f"*****HOOOMAN DONE*************")

    # mouse_input_path = "/work/suchanda/ligand_receptor_mining/src/celltalkdb/celltalk_mouse_pubmed.csv"
    # prepare_dataset(mouse_input_path, organism_name="mouse")
    # print(f"*****MOUSE DONE*************")


if __name__ == "__main__":
    query = "kidney ligand receptors"
    run(query = query)
    
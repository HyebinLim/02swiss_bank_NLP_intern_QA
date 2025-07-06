#abstract all of this into a function that takes in a PDF file name 

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
import streamlit as st

def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    try:
        # load documents with progress indicator
        with st.spinner("Loading PDF document..."):
            documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        
        with st.spinner("Processing document chunks..."):
            # Optimized chunk size for better performance and accuracy
            splitter = SentenceSplitter(
                chunk_size=1024,  # Larger chunks for better context
                chunk_overlap=100,  # More overlap for better continuity
                separator="\n"  # Use newlines as separators
            )
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Filter out very short nodes (likely noise)
            filtered_nodes = [node for node in nodes if len(node.text.strip()) > 50]
            
            # Limit number of nodes for better performance
            if len(filtered_nodes) > 800:
                filtered_nodes = filtered_nodes[:800]
                st.warning("Document was optimized to 800 chunks for better performance.")
            
            nodes = filtered_nodes
        
        with st.spinner("Creating vector index..."):
            vector_index = VectorStoreIndex(nodes)
        
        def vector_query(
            query: str, 
            page_numbers: Optional[List[str]] = None
        ) -> str:
            """Use to answer questions over the posts.
        
            Useful if you have specific questions over the posts.
            Always leave page_numbers as None UNLESS there is a specific page(in pdf) you want to search for.
        
            Args:
                query (str): the string query to be embedded.
                page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                    if we want to perform a vector search
                    over all pages. Otherwise, filter by the set of specified pages.
            
            """
        
            page_numbers = page_numbers or []
            metadata_dicts = [
                {"key": "page_label", "value": p} for p in page_numbers
            ]
            
            query_engine = vector_index.as_query_engine(
                similarity_top_k=3,  # More relevant results
                filters=MetadataFilters.from_dicts(
                    metadata_dicts,
                    condition=FilterCondition.OR
                ) if metadata_dicts else None,
                response_mode="compact",  # More concise responses
                streaming=False  # Disable streaming for better performance
            )
            response = query_engine.query(query)
            return response
            
        
        vector_query_tool = FunctionTool.from_defaults(
            name=f"vector_tool_{name}",
            fn=vector_query
        )
        
        with st.spinner("Creating summary index..."):
            summary_index = SummaryIndex(nodes)
            summary_query_engine = summary_index.as_query_engine(
                response_mode="tree_summarize",
                use_async=True,
            )
            summary_tool = QueryEngineTool.from_defaults(
                name=f"summary_tool_{name}",
                query_engine=summary_query_engine,
                description=(
                    "Use ONLY IF you want a summary."
                    "Do NOT use if you have specific questions."
                ),
            )

        return vector_query_tool, summary_tool
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        raise e
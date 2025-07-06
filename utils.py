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
            # Smaller chunk size for better performance
            splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Limit number of nodes for better performance
            if len(nodes) > 1000:
                nodes = nodes[:1000]
                st.warning("Document was truncated to 1000 chunks for better performance.")
        
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
                similarity_top_k=2,
                filters=MetadataFilters.from_dicts(
                    metadata_dicts,
                    condition=FilterCondition.OR
                )
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
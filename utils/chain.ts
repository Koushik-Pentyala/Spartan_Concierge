import { OpenAI } from "langchain/llms/openai";
import { pinecone } from "@/utils/pinecone-client";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { ConversationalRetrievalQAChain } from "langchain/chains";

async function initChain() {
    console.log("Initializing OpenAI model...");
    const model = new OpenAI({});
    console.log("OpenAI model initialized.");

    console.log("Retrieving Pinecone index...");
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX ?? '');
    console.log(`Pinecone index retrieved: ${process.env.PINECONE_INDEX}`);

    try {
        console.log("Creating vector store from existing index...");
        const vectorStore = await PineconeStore.fromExistingIndex(
            new OpenAIEmbeddings({}),
            {
                pineconeIndex: pineconeIndex,
                textKey: 'text',
            },
        );
        console.log("Vector store created successfully.");

        console.log("Initializing ConversationalRetrievalQAChain...");
        const chain = ConversationalRetrievalQAChain.fromLLM(
            model,
            vectorStore.asRetriever(),
            { returnSourceDocuments: true }
        );
        console.log("ConversationalRetrievalQAChain initialized successfully.");

        return chain;
    } catch (error) {
        console.error("Error creating vector store or initializing QA chain:", error);
        throw error;
    }
}

export const chain = await initChain();

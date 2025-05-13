from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

class ClassifierOutput(BaseModel):
    """
    Data structure for the model's output.
    """

    category: list = Field(
        description="A list of clothes category to search for ('Tshirts', 'Jeans', 'Track Pants', 'shoes', 'jersey', 'Shirts')."
    )
    number_of_items: int = Field(description="The number of items we should retrieve. Minimum 5, maximum 10.")

class Classifier:
    """
    Classifier class for classification of input text.
    """
    
    def __init__(self, model: ChatGoogleGenerativeAI) -> None:
        """
        Initialize the Chain class by creating the chain.
        Args:
            model (ChatVertexAI): The LLM model.
        """
        super().__init__()

        parser = PydanticOutputParser(pydantic_object=ClassifierOutput)

        articles= ['Shirts', 'Jeans', 'Watches', 'Track Pants', 'Tshirts', 'Socks',
       'Casual Shoes', 'Belts', 'Flip Flops', 'Handbags', 'Tops', 'Bra',
       'Sandals', 'Shoe Accessories', 'Sweatshirts', 'Deodorant',
       'Formal Shoes', 'Bracelet', 'Lipstick', 'Flats', 'Kurtas',
       'Waistcoat', 'Sports Shoes', 'Shorts', 'Briefs', 'Sarees',
       'Perfume and Body Mist', 'Heels', 'Sunglasses', 'Innerwear Vests',
       'Pendant', 'Nail Polish', 'Laptop Bag', 'Scarves', 'Rain Jacket',
       'Dresses', 'Night suits', 'Skirts', 'Wallets', 'Blazers', 'Ring',
       'Kurta Sets', 'Clutches', 'Shrug', 'Backpacks', 'Caps', 'Trousers',
       'Earrings', 'Camisoles', 'Boxers', 'Jewellery Set', 'Dupatta',
       'Capris', 'Lip Gloss', 'Bath Robe', 'Mufflers', 'Tunics',
       'Jackets', 'Trunk', 'Lounge Pants', 'Face Wash and Cleanser',
       'Necklace and Chains', 'Duffel Bag', 'Sports Sandals',
       'Foundation and Primer', 'Sweaters', 'Free Gifts', 'Trolley Bag',
       'Tracksuits', 'Swimwear', 'Shoe Laces', 'Fragrance Gift Set',
       'Bangle', 'Nightdress', 'Ties', 'Baby Dolls', 'Leggings',
       'Highlighter and Blush', 'Travel Accessory', 'Kurtis',
       'Mobile Pouch', 'Messenger Bag', 'Lip Care', 'Face Moisturisers',
       'Compact', 'Eye Cream', 'Accessory Gift Set', 'Beauty Accessory',
       'Jumpsuit', 'Kajal and Eyeliner', 'Water Bottle', 'Suspenders',
       'Lip Liner', 'Robe', 'Salwar and Dupatta', 'Patiala', 'Stockings',
       'Eyeshadow', 'Headband', 'Tights', 'Nail Essentials', 'Churidar',
       'Lounge Tshirts', 'Face Scrub and Exfoliator', 'Lounge Shorts',
       'Gloves', 'Mask and Peel', 'Wristbands', 'Tablet Sleeve',
       'Ties and Cufflinks', 'Footballs', 'Stoles', 'Shapewear',
       'Nehru Jackets', 'Salwar', 'Cufflinks', 'Jeggings', 'Hair Colour',
       'Concealer', 'Rompers', 'Body Lotion', 'Sunscreen', 'Booties',
       'Waist Pouch', 'Hair Accessory', 'Rucksacks', 'Basketballs',
       'Lehenga Choli', 'Clothing Set', 'Mascara', 'Toner',
       'Cushion Covers', 'Key chain', 'Makeup Remover', 'Lip Plumper',
       'Umbrellas', 'Face Serum and Gel', 'Hat', 'Mens Grooming Kit',
       'Rain Trousers', 'Body Wash and Scrub', 'Suits', 'Ipad']

        text_prompt = """
        You are a fashion assistant expert on understanding what a customer needs and on extracting the category or categories of clothes a customer wants from the given text.
        Text:
        {text}

        Instructions:
        1. Read carefully the text.
        2. Extract the category or categories of clothes the customer is looking for, it can be:
            - Tshirts if the custimer is looking for a t-shirt.
            - Jeans if the customer is looking for jeans.
            - Track Pants if the customer is looking for pants.
            - jacket if the customer is looking for a jacket.
            - shoes if the customer is looking for shoes.
            - jersey if the customer is looking for a jersey.
            - Shirts if the customer is looking for a shirt.
        3. If the customer is looking for multiple items of the same category, return the number of items we should retrieve. If not specfied but the user asked for more than 1, return 5.
        4. If the customer is looking for multiple category, the number of items should be 1.
        5. Return a valid JSON with the categories found, the key must be 'category' and the value must be a list with the categories found and 'number_of_items' with the number of items we should retrieve.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        text_prompt = """
        You are a fashion assistant expert on understanding what a customer needs and on extracting the category or categories of clothes a customer wants from the given text.
        Text:
        {text}

        Instructions:
        1. Read carefully the text.
        2. Extract the category or categories of items the customer is looking for, the categories can be any of the following:
            {articles}

        3. If the customer is looking for multiple items of the same category, return the number of items we should retrieve. If not specfied but the user asked for more than 1, return 5.
        4. If the customer is looking for multiple category, the number of items should be 1.
        5. Return a valid JSON with the categories found, the key must be 'category' and the value must be a list with the categories found and 'number_of_items' with the number of items we should retrieve.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        prompt = PromptTemplate.from_template(
            text_prompt, partial_variables={"format_instructions": parser.get_format_instructions(), "articles":articles}
        )
        self.chain = prompt | model | parser

    def classify(self, text: str) -> ClassifierOutput:
        """
        Get the category from the model based on the text context.
        Args:
            text (str): user message.
        Returns:
            ClassifierOutput: The model's answer.
        """
        try:
            return self.chain.invoke({"text": text})
        except Exception as e:
            raise RuntimeError(f"Error invoking the chain: {e}")

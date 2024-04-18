import re
import json
import string
import contractions

"""
Process and clean text data from a JSON file specified by:

1. reading `file_path` where each line is considered a separate JSON object
2. cleaning and normalizing each JSON object
3. saving that cleaned JSON back into a file called, by default, "data.json"

Args:
    file_path (str): Path to the input JSON file containing the raw text data.
    output_path (str) *optional: Desired name for the output file

Outputs:
    data.json (file): A JSON file of cleaned text
"""
def process_file(file_path, output_path=None):

    # Read the JSON file and parse each line into a dictionary
    json_objects = []
    with open(file_path, 'r') as txt:
        for jsonObj in txt:
            dict = json.loads(jsonObj)
            json_objects.append(dict)

    # Process text of each JSON object
    for obj in json_objects:
        obj_text = obj["text"].lower()

        # Form contractions naively
        obj_text = re.sub("I m", 'I\'m', obj_text)
        obj_text = re.sub("i m", 'i\'m', obj_text)
        obj_text = re.sub("Don t", 'Don\'t', obj_text)
        obj_text = re.sub("don t", 'don\'t', obj_text)
        obj_text = re.sub("I ll", 'I\'ll', obj_text)
        obj_text = re.sub("i ll", 'i\'ll', obj_text)

        # Expand correctly formed contractions
        expanded_txt = [] 
        for word in obj_text.split():
            expanded_txt.append(contractions.fix(word))  
        obj_text = ' '.join(expanded_txt)

        # Break apart incorrectly formed contractions
        obj_text = re.sub("\'", ' ', obj_text).lstrip()

        # Remove all punctuation
        obj_text = re.sub("“", "", obj_text)
        obj_text = re.sub("”", "", obj_text)
        obj_text = re.sub("‘", "", obj_text)
        obj_text = re.sub("’", "", obj_text)
        obj_text = re.sub(r"[{}]".format(string.punctuation.replace("#", "")), "", obj_text)

        obj_text = re.sub(r"\s*{.*}\s*", " ", obj_text)                          # Remove unknown mentions and links
        obj_text = re.sub(r"\s*@.*[A-z]\s*", " ", obj_text)                      # Remove rest of mentions
        obj_text = re.sub(r"\s+", ' ', obj_text).lstrip()                        # Remove repeated spaces and leading whitespace

        obj["text"] = obj_text

    print(json_objects)

    if output_path == None: output_file_path = "data.json"
    else: output_file_path = output_path

    with open(output_file_path, "w") as outfile:
        json.dump(json_objects, outfile)

#process_file("tweet_topic_single/dataset/split_coling2022_random/test_random.single.json")
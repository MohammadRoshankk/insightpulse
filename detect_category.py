def detect_category(text):
    complaint =0
    feature_request = 0
    general =0

    complaint_phrases = ["broke", "broken", "damaged", "damage", "stopped working", 
                      "cheaply", "poor", "defective", "doesn't work", "faulty", 
                      "malfunctioning", "cracked", "leaking", "issue", "problem"]
    feature_request_phrases = ["wish it had", "would be nice if", "should include", "needs a", "hope they add"]
    
    for word in complaint_phrases:
        if word in text:
            complaint+=1
    
    for word in feature_request_phrases:
        if word in text:
            feature_request+=1
    
    if complaint>feature_request:
        return "complaint"
    elif feature_request>complaint:
        return "feature request"
    else:
        return "general"
    


#   - Only detects exact phrases
#   - No context understanding
#   - Cannot handle synonyms or variations
#   - Might misclassify if keywords appear in unrelated contexts
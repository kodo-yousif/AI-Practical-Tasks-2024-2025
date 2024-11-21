word_list = [
    "apple", "application", "apply", "ape", "apricot",
    "banana", "bandana", "grape", "pineapple", "peach",
    "pear", "plum", "pomegranate", "mango", "melon",
    "kiwi", "blueberry", "blackberry", "cranberry", "cherry",
    "date", "fig", "guava", "lemon", "lime",
    "orange", "papaya", "passionfruit", "raspberry", "strawberry",
    "watermelon", "coconut", "dragonfruit", "elderberry", "gooseberry",
    "lychee", "nectarine", "quince", "tangerine", "cucumber",
    "pumpkin", "zucchini", "avocado", "persimmon", "kumquat",
    "mushroom", "onion", "tomato", "potato", "carrot",
    "cabbage", "cauliflower", "broccoli", "spinach", "lettuce",
    "kale", "asparagus", "celery", "radish", "turnip",
    "garlic", "ginger", "basil", "parsley", "thyme",
    "oregano", "rosemary", "coriander", "dill", "chives",
    "pistachio", "cashew", "almond", "peanut", "walnut",
    "hazelnut", "pecan", "macadamia", "brazilnut", "chestnut",
    # Fruits
    "apple", "application", "apply", "ape", "apricot", "banana", "bandana", "grape", "pineapple", "peach",
    "pear", "plum", "pomegranate", "mango", "melon", "kiwi", "blueberry", "blackberry", "cranberry", "cherry",
    "date", "fig", "guava", "lemon", "lime", "orange", "papaya", "passionfruit", "raspberry", "strawberry",
    "watermelon", "coconut", "dragonfruit", "elderberry", "gooseberry", "lychee", "nectarine", "quince", "tangerine",

    # Vegetables
    "cucumber", "pumpkin", "zucchini", "avocado", "persimmon", "kumquat", "mushroom", "onion", "tomato", "potato",
    "carrot", "cabbage", "cauliflower", "broccoli", "spinach", "lettuce", "kale", "asparagus", "celery", "radish",
    "turnip", "garlic", "ginger", "basil", "parsley", "thyme", "oregano", "rosemary", "coriander", "dill",

    # Herbs & Spices
    "chives", "pistachio", "cashew", "almond", "peanut", "walnut", "hazelnut", "pecan", "macadamia", "brazilnut",
    "chestnut", "saffron", "vanilla", "cinnamon", "clove", "nutmeg", "paprika", "peppermint", "cayenne", "turmeric",

    # Grains & Cereals
    "rice", "quinoa", "barley", "millet", "oats", "wheat", "buckwheat", "sorghum", "amaranth", "rye",
    "corn", "polenta", "bulgur", "couscous", "farro", "spelt", "teff",

    # Dairy Products
    "bread", "butter", "cheese", "yogurt", "milk", "cream", "custard", "gelato", "sherbet", "pudding",
    "mozzarella", "parmesan", "cheddar", "brie", "feta", "ricotta", "gouda", "camembert",

    # Desserts & Sweets
    "cake", "cookie", "brownie", "muffin", "cupcake", "donut", "pie", "tart", "pastry", "croissant",
    "macaron", "eclair", "pancake", "waffle", "gelatin", "jelly", "pudding", "candy", "toffee", "caramel",

    # Drinks & Beverages
    "coffee", "tea", "cocoa", "juice", "soda", "water", "milkshake", "smoothie", "lemonade", "icedtea",
    "whiskey", "vodka", "rum", "gin", "tequila", "wine", "beer", "cider", "champagne", "brandy",

    # Technology & Programming
    "algorithm", "binary", "compiler", "debugger", "database", "frontend", "backend", "middleware", "cloud", "container",
    "python", "javascript", "typescript", "react", "nodejs", "django", "flask", "fastapi", "graphql", "restapi",
    "html", "css", "json", "xml", "http", "https", "tcp", "udp", "sql", "nosql",

    # Common Verbs
    "run", "jump", "swim", "read", "write", "code", "sing", "dance", "eat", "sleep",
    "drive", "fly", "draw", "paint", "build", "create", "destroy", "explore", "learn", "teach",

    # Common Adjectives
    "happy", "sad", "angry", "excited", "nervous", "calm", "brave", "shy", "lucky", "unlucky",
    "bright", "dark", "light", "heavy", "soft", "hard", "hot", "cold", "warm", "cool",

    # Miscellaneous
    "universe", "galaxy", "planet", "earth", "moon", "star", "comet", "asteroid", "meteor", "blackhole",
    "ocean", "sea", "river", "lake", "pond", "waterfall", "mountain", "valley", "desert", "forest",
    "city", "village", "town", "country", "state", "continent", "island", "peninsula", "bay", "coast",
    "school", "university", "library", "museum", "theater", "hospital", "airport", "station", "park", "zoo",
    "phone", "laptop", "keyboard", "mouse", "monitor", "tablet", "smartwatch", "speaker", "camera", "microphone",

    "rice", "quinoa", "barley", "millet", "oats",
    "wheat", "buckwheat", "sorghum", "amaranth", "rye",
    "bread", "butter", "cheese", "yogurt", "milk",
    "cream", "custard", "gelato", "sherbet", "pudding",
    "cake", "cookie", "brownie", "muffin", "cupcake",
    "pizza", "burger", "sandwich", "taco", "burrito",
    "sushi", "noodle", "pasta", "spaghetti", "lasagna",
    "ravioli", "dumpling", "ramen", "pho", "soba",
    "tofu", "seitan", "tempeh", "edamame", "miso",
    "curry", "stew", "soup", "broth", "bisque",
    "salsa", "guacamole", "hummus", "tahini", "tzatziki",
    "sugar", "salt", "pepper", "vinegar", "oil",
    "butter", "margarine", "mayonnaise", "ketchup", "mustard",
    "jam", "jelly", "honey", "syrup", "molasses",
    "coffee", "tea", "cocoa", "juice", "soda",
    "water", "milkshake", "smoothie", "lemonade", "icedtea",
    "whiskey", "vodka", "rum", "gin", "tequila",
    "wine", "beer", "cider", "champagne", "brandy",
<<<<<<< HEAD
    "mojito", "margarita", "martini", "sangria", "pina_colada", "fake madrid",
=======
    "mojito", "margarita", "martini", "sangria", "pina_colada"
>>>>>>> 7db2bcd (Group E - Ahmed Adnan - Auto Correction/Completion (#27))
]
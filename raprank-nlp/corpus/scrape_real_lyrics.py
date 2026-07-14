import os
import re
import json
import sys
import time
from pathlib import Path
import httpx
from bs4 import BeautifulSoup
from mistralai import Mistral
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

API_KEY = os.getenv("MISTRAL_API_KEY")
GENIUS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
DATA_DIR = Path(__file__).resolve().parent / "data"

# Artists that were generated using LLM lyrics
TARGET_ARTISTS = [
    {
        "name": "EPR",
        "dir": "epr",
        "genius_id": 2099093,
        "lang": "mixed",
        "tracks": {
            "Badluram ka badan":   "https://genius.com/Epr-badluram-ka-badan-lyrics",
            "Raastamahn":          "https://genius.com/Epr-raastamahn-lyrics",
            "Ekla Cholo Re":       "https://genius.com/Epr-ekla-cholo-re-lyrics",
            "Srini Bana EPR":      "https://genius.com/Epr-srini-bana-epr-lyrics",
            "Q":                   "https://genius.com/Epr-iyer-q-lyrics",
            "ADR / ABR":           "https://genius.com/Epr-adr-abr-lyrics",
            "Koi Gham Nahi":       "https://genius.com/Epr-koi-gham-nahi-lyrics",
            "Fibonacci":           "https://genius.com/Epr-fibonacci-lyrics",
        }
    },
    {
        "name": "Naam Sujal",
        "dir": "naam-sujal",
        "genius_id": 2780157,
        "lang": "mixed",
        "tracks": {
            "The Waddup":          "https://genius.com/Naam-sujal-the-waddup-lyrics",
            "Blind Spot":          "https://genius.com/Naam-sujal-blind-spot-lyrics",
            "Protocol":            "https://genius.com/Naam-sujal-protocol-lyrics",
            "PYAAR?":              "https://genius.com/Naam-sujal-pyaar-lyrics",
            "Its About Time":      "https://genius.com/Naam-sujal-its-about-time-lyrics",
            "Blueprint":           "https://genius.com/Naam-sujal-blueprint-lyrics",
            "Vishay Khatam":       "https://genius.com/Naam-sujal-vishay-khatam-lyrics",
            "Dafli Wale":          "https://genius.com/Naam-sujal-dafli-wale-lyrics",
        }
    },
    {
        "name": "Vichaar",
        "dir": "vichaar",
        "genius_id": 3815901,
        "lang": "mixed",
        "tracks": {
            "3 DRAGS":                  "https://genius.com/Vichaar-3-drags-lyrics",
            "Kalakaari Vishwasniya":    "https://genius.com/Vichaar-kalakaari-vishwasniya-lyrics",
            "Mudda Kya Hai":            "https://genius.com/Naam-sujal-and-vichaar-mudda-kya-hai-lyrics",
            "5 Fingers of Death":       "https://genius.com/Vichaar-5-fingers-of-death-lyrics",
            "Haadse":                   "https://genius.com/Vichaar-haadse-lyrics",
            "No Pockets":               "https://genius.com/Vichaar-no-pockets-lyrics",
            "Bhul":                     "https://genius.com/Vichaar-bhul-lyrics",
            "Mehnat":                   "https://genius.com/Vichaar-mehnat-lyrics",
        }
    },
    {
        "name": "Lil Bhatia",
        "dir": "lil-bhatia",
        "genius_id": 3495224,
        "lang": "mixed",
        "tracks": {
            "Taakat":       "https://genius.com/Seedhe-maut-taakat-lyrics",
            "Maar Kaat":    "https://genius.com/Seedhe-maut-maar-kaat-lyrics",
            "Peace of Mind":"https://genius.com/Seedhe-maut-peace-of-mind-lyrics",
        }
    },
    {
        "name": "Yungsta",
        "dir": "yungsta",
        "genius_id": 174215,
        "lang": "mixed",
        "tracks": {
            "Ruhbaru":          "https://genius.com/Yungsta-ruhbaru-lyrics",
            "Dilli":            "https://genius.com/Yungsta-dilli-lyrics",
            "Sansani":          "https://genius.com/Yungsta-sansani-lyrics",
            "Sukoon":           "https://genius.com/Yungsta-sukoon-lyrics",
            "Savera":           "https://genius.com/Yungsta-savera-lyrics",
            "Totka":            "https://genius.com/Yungsta-totka-lyrics",
            "Hona Hi Tha":      "https://genius.com/Yungsta-hona-hi-tha-lyrics",
            "Kaamyaabi":        "https://genius.com/Yungsta-kaamyaabi-lyrics",
            "Jeena Isi Ka Naam":"https://genius.com/Yungsta-jeena-isi-ka-naam-lyrics",
        }
    },
    {
        "name": "Raga",
        "dir": "raga",
        "genius_id": 1050178,
        "lang": "mixed",
        "tracks": {
            "Rap Ka Mausam": "https://genius.com/Raga-rap-ka-mausam-lyrics",
            "Sheher":        "https://genius.com/Raga-sheher-lyrics",
            "Jamnapaar":     "https://genius.com/Raga-jamnapaar-lyrics",
        }
    },
    {
        "name": "Panther",
        "dir": "panther",
        "genius_id": 3354305,
        "lang": "mixed",
        "tracks": {
            "Galat Karam":   "https://genius.com/Panther-ind-galat-karam-lyrics",
            "Parinda":       "https://genius.com/Panther-ind-parinda-lyrics",
            "Oh My God":     "https://genius.com/Panther-ind-oh-my-god-lyrics",
            "Rangey Haath":  "https://genius.com/Panther-ind-rangey-haath-lyrics",
            "Rukna Nahi Tha":"https://genius.com/Panther-ind-rukna-nahi-tha-lyrics",
            "Aisi Jagah Se": "https://genius.com/Panther-ind-aisi-jagah-se-lyrics",
            "Sajke":         "https://genius.com/Panther-ind-sajke-lyrics",
            "Bemisaal":      "https://genius.com/Panther-ind-bemisaal-lyrics",
        }
    },
    {
        "name": "Paradox",
        "dir": "paradox",
        "genius_id": 3509765,
        "lang": "mixed",
        "tracks": {
            "Jaadugar":      "https://genius.com/Paradox-jaadugar-lyrics",
            "Glitch":        "https://genius.com/Paradox-and-ishh-glitch-lyrics",
            "Ghatotkach":    "https://genius.com/Paradox-and-pinnocio-ghatotkach-lyrics",
            "Hasti Rahe Tu": "https://genius.com/Paradox-hasti-rahe-tu-lyrics",
            "Zimmedaar":     "https://genius.com/Paradox-and-aod-ind-zimmedaar-lyrics",
        }
    },
    {
        "name": "Tsumyoki",
        "dir": "tsumyoki",
        "genius_id": 1691777,
        "lang": "mixed",
        "tracks": {
            "Pink Blue":            "https://genius.com/Tsumyoki-and-bharg-pink-blue-lyrics",
            "Ek Do Ek":             "https://genius.com/Tsumyoki-and-rawal-ek-do-ek-lyrics",
            "Perfect Life":         "https://genius.com/Tsumyoki-perfect-life-lyrics",
            "WHAT CAN I SAY?":      "https://genius.com/Tsumyoki-and-arpit-bala-what-can-i-say-lyrics",
            "BREAKSHIT!":           "https://genius.com/Tsumyoki-breakshit-lyrics",
            "WANT IT ALL":          "https://genius.com/Tsumyoki-and-mc-square-want-it-all-lyrics",
            "Dont Even Text":       "https://genius.com/Tsumyoki-and-gini-ind-dont-even-text-lyrics",
            "Sunlight":             "https://genius.com/Tsumyoki-sunlight-lyrics",
            "The Way I Fall In Love":"https://genius.com/Tsumyoki-the-way-i-fall-in-love-lyrics",
            "Its Aight":            "https://genius.com/Tsumyoki-its-aight-lyrics",
            "Headphones":           "https://genius.com/Tsumyoki-headphones-lyrics",
            "IDK":                  "https://genius.com/Tsumyoki-idk-lyrics",
            "Candyland":            "https://genius.com/Tsumyoki-candyland-lyrics",
            "On My Way":            "https://genius.com/Tsumyoki-and-elttwo-on-my-way-lyrics",
            "Feel Okay":            "https://genius.com/Tsumyoki-feel-okay-lyrics",
        }
    },
    # ─── Expanded existing artists ───────────────────────────────────────────
    {
        "name": "Raga",
        "dir": "raga",
        "genius_id": 1050178,
        "lang": "mixed",
        "tracks": {
            "Rap Ka Mausam":    "https://genius.com/Raga-rap-ka-mausam-lyrics",
            "Sheher":           "https://genius.com/Raga-sheher-lyrics",
            "Jamnapaar":        "https://genius.com/Raga-jamnapaar-lyrics",
            "Badan":            "https://genius.com/Raga-badan-lyrics",
            "Jahannum":         "https://genius.com/Raga-jahannum-lyrics",
            "Skills":           "https://genius.com/Raga-skills-lyrics",
            "GTA NCR":          "https://genius.com/Raga-and-wamp-gta-ncr-lyrics",
            "Akatsuki":         "https://genius.com/Seedhe-maut-and-raga-akatsuki-lyrics",
        }
    },
    {
        "name": "Panther",
        "dir": "panther",
        "genius_id": 3354305,
        "lang": "mixed",
        "tracks": {
            "Galat Karam":      "https://genius.com/Panther-ind-galat-karam-lyrics",
            "Parinda":          "https://genius.com/Panther-ind-parinda-lyrics",
            "Oh My God":        "https://genius.com/Panther-ind-oh-my-god-lyrics",
            "Rangey Haath":     "https://genius.com/Panther-ind-rangey-haath-lyrics",
            "Rukna Nahi Tha":   "https://genius.com/Panther-ind-rukna-nahi-tha-lyrics",
            "Aisi Jagah Se":    "https://genius.com/Panther-ind-aisi-jagah-se-lyrics",
            "Sajke":            "https://genius.com/Panther-ind-sajke-lyrics",
            "Bemisaal":         "https://genius.com/Panther-ind-bemisaal-lyrics",
            "Non Talha":        "https://genius.com/Panther-ind-non-talha-lyrics",
            "Kiska Hai":        "https://genius.com/Panther-ind-kiska-hai-lyrics",
            "Rang":             "https://genius.com/Panther-ind-rang-lyrics",
            "Bohot Aagey":      "https://genius.com/Panther-ind-bohot-aagey-lyrics",
            "Jaani":            "https://genius.com/Panther-ind-jaani-lyrics",
            "Parwah":           "https://genius.com/Panther-ind-parwah-lyrics",
        }
    },
    {
        "name": "Seedhe Maut",
        "dir": "seedhe-maut",
        "genius_id": 1783426,
        "lang": "mixed",
        "tracks": {
            "Nanchaku":         "https://genius.com/Seedhe-maut-nanchaku-lyrics",
            "Namastute":        "https://genius.com/Seedhe-maut-namastute-lyrics",
            "Khatta Flow":      "https://genius.com/Seedhe-maut-and-kr-na-khatta-flow-lyrics",
            "Alla Freestyle":   "https://genius.com/Seedhe-maut-and-dj-sa-alla-freestyle-lyrics",
            "Shakti Aur Kshama":"https://genius.com/Seedhe-maut-shakti-aur-kshama-lyrics",
            "101":              "https://genius.com/Seedhe-maut-101-lyrics",
            "11k":              "https://genius.com/Seedhe-maut-11k-lyrics",
            "MMM":              "https://genius.com/Seedhe-maut-mmm-lyrics",
            "Batti":            "https://genius.com/Seedhe-maut-and-sez-on-the-beat-batti-lyrics",
            "Maina":            "https://genius.com/Seedhe-maut-and-sez-on-the-beat-maina-lyrics",
            "Do Guna":          "https://genius.com/Seedhe-maut-do-guna-lyrics",
            "Kohra":            "https://genius.com/Seedhe-maut-and-sez-on-the-beat-kohra-lyrics",
            "Nawazuddin":       "https://genius.com/Seedhe-maut-nawazuddin-lyrics",
            "Naksha":           "https://genius.com/Seedhe-maut-naksha-lyrics",
            "Natkhat":          "https://genius.com/Seedhe-maut-natkhat-lyrics",
            "Taakat":           "https://genius.com/Seedhe-maut-taakat-lyrics",
            "Maar Kaat":        "https://genius.com/Seedhe-maut-maar-kaat-lyrics",
            "Peace of Mind":    "https://genius.com/Seedhe-maut-peace-of-mind-lyrics",
            "Raat Ki Raani":    "https://genius.com/Seedhe-maut-raat-ki-raani-lyrics",
            "Brahamachari":     "https://genius.com/Seedhe-maut-brahamachari-lyrics",
            "Gourmet Shit":     "https://genius.com/Seedhe-maut-gourmet-shit-lyrics",
        }
    },
    {
        "name": "JTrix",
        "dir": "jtrix",
        "genius_id": 1836940,
        "lang": "mixed",
        "tracks": {
            "Bohot Sahi":       "https://genius.com/J-trix-bohot-sahi-lyrics",
            "Khayaal":          "https://genius.com/J-trix-and-subspace-khayaal-lyrics",
            "Tehelka":          "https://genius.com/J-trix-and-subspace-tehelka-lyrics",
            "Muskura":          "https://genius.com/J-trix-and-subspace-muskura-lyrics",
            "Zindagi":          "https://genius.com/J-trix-and-subspace-zindagi-lyrics",
            "Pehchaan":         "https://genius.com/J-trix-and-subspace-pehchaan-lyrics",
            "Jazbaat":          "https://genius.com/J-trix-and-subspace-jazbaat-lyrics",
            "Sharaab":          "https://genius.com/J-trix-sharaab-lyrics",
            "Ek Tarfa":         "https://genius.com/J-trix-and-subspace-ek-tarfa-lyrics",
            "Kya Scene":        "https://genius.com/J-trix-and-subspace-kya-scene-lyrics",
            "Streets":          "https://genius.com/J-trix-streets-lyrics",
            "Masti Nahi":       "https://genius.com/J-trix-masti-nahi-lyrics",
            "Blow Up":          "https://genius.com/J-trix-blow-up-lyrics",
            "Kalaakaar":        "https://genius.com/J-trix-and-subspace-kalaakaar-lyrics",
            "Mera Naam":        "https://genius.com/J-trix-and-subspace-mera-naam-lyrics",
            "One and Only":     "https://genius.com/J-trix-one-and-only-lyrics",
            "Kaala":            "https://genius.com/J-trix-kaala-lyrics",
            "Sort Hai":         "https://genius.com/J-trix-sort-hai-lyrics",
            "Mast":             "https://genius.com/J-trix-and-subspace-mast-lyrics",
            "Banger Pro Max":   "https://genius.com/J-trix-banger-pro-max-lyrics",
        }
    },
    {
        "name": "Muhfaad",
        "dir": "muhfaad",
        "genius_id": 1276527,
        "lang": "mixed",
        "tracks": {
            "Bhoot Banega":         "https://genius.com/Muhfaad-bhoot-banega-lyrics",
            "Aelaan":               "https://genius.com/Muhfaad-aelaan-lyrics",
            "Backflip":             "https://genius.com/Muhfaad-backflip-lyrics",
            "Happy Diwala Ravan":   "https://genius.com/Muhfaad-happy-diwala-ravan-lyrics",
            "Sach Too Much Hai":    "https://genius.com/Muhfaad-sach-too-much-hai-lyrics",
            "Bakri (GOAT)":         "https://genius.com/Muhfaad-bakri-goat-lyrics",
            "Maa Kasam":            "https://genius.com/Muhfaad-maa-kasam-lyrics",
            "Nateejay":             "https://genius.com/Muhfaad-and-kartavya-nateejay-lyrics",
            "Jaldi Aao":            "https://genius.com/Muhfaad-jaldi-aao-lyrics",
            "Ambaran":              "https://genius.com/Muhfaad-ambaran-lyrics",
            "Adayein":              "https://genius.com/Muhfaad-adayein-lyrics",
            "Har Har Gange":        "https://genius.com/Muhfaad-har-har-gange-lyrics",
            "Duniya Chutiya":       "https://genius.com/Muhfaad-duniya-chutiya-lyrics",
            "Ego Friendly":         "https://genius.com/Muhfaad-ego-friendly-lyrics",
            "Abe Chodna":           "https://genius.com/Emiway-bantai-and-muhfaad-abe-chodna-lyrics",
            "Sare Karo Dab":        "https://genius.com/Raftaar-sonu-kakkar-and-muhfaad-sare-karo-dab-lyrics",
            "Har Har Gange":        "https://genius.com/Muhfaad-har-har-gange-lyrics",
            "Glitch":               "https://genius.com/Muhfaad-glitch-lyrics",
            "Kali Zuban":           "https://genius.com/Muhfaad-kali-zuban-lyrics",
            "Instagram Live":       "https://genius.com/Muhfaad-instagram-live-lyrics",
            "Moksh":                "https://genius.com/Muhfaad-moksh-lyrics",
        }
    },
    {
        "name": "KR$NA",
        "dir": "kr-na",
        "genius_id": 1562459,
        "lang": "mixed",
        "tracks": {
            # Previously in corpus (lyricsmint-scraped, now restored via Genius)
            "10 Pe 10":             "https://genius.com/Kr-na-10-pe-10-lyrics",
            "Been A While":         "https://genius.com/Kr-na-been-a-while-lyrics",
            "Blowing Up":           "https://genius.com/Kr-na-blowing-up-lyrics",
            "Hello":               "https://genius.com/Kr-na-hello-lyrics",
            "Lil Bunty":            "https://genius.com/Kr-na-lil-bunty-lyrics",
            "Makasam":              "https://genius.com/Kr-na-makasam-lyrics",
            "Muqabla":              "https://genius.com/Kr-na-muqabla-lyrics",
            "OG":                   "https://genius.com/Kr-na-og-lyrics",
            "Roll Up":              "https://genius.com/Kr-na-roll-up-lyrics",
            "Saath Ya Khilaaf":     "https://genius.com/Kr-na-saath-ya-khilaaf-lyrics",
            "Saza E Maut":          "https://genius.com/Kr-na-saza-e-maut-lyrics",
            "Shut Up":              "https://genius.com/Kr-na-shut-up-lyrics",
            "Some Of Us":           "https://genius.com/Kr-na-some-of-us-lyrics",
            "Still Standing":       "https://genius.com/Kr-na-still-standing-lyrics",
            "Villain":              "https://genius.com/Kr-na-villain-lyrics",
            "Wanna Know":           "https://genius.com/Kr-na-wanna-know-lyrics",
            "Whats Up":             "https://genius.com/Kr-na-whats-up-lyrics",
            # New Genius additions
            "I Guess":              "https://genius.com/Kr-na-i-guess-lyrics",
            "Untitled":             "https://genius.com/Kr-na-untitled-lyrics",
            "No Cap":               "https://genius.com/Kr-na-no-cap-lyrics",
            "Machayenge 4":         "https://genius.com/Kr-na-machayenge-4-lyrics",
            "Seedha Makeover":      "https://genius.com/Kr-na-seedha-makeover-lyrics",
            "Prarthana":            "https://genius.com/Kr-na-and-bharg-prarthana-lyrics",
            "Vyanjan":              "https://genius.com/Kr-na-vyanjan-hindi-alphabetic-rap-lyrics",
            "Knock Knock":          "https://genius.com/Kr-na-knock-knock-lyrics",
            "NGL":                  "https://genius.com/Kr-na-ngl-lyrics",
            "Maharani":             "https://genius.com/Kr-na-maharani-lyrics",
            "Say My Name":          "https://genius.com/Kr-na-say-my-name-english-version-lyrics",
            "Sensitive":            "https://genius.com/Kr-na-sensitive-lyrics",
            "Joota Japani":         "https://genius.com/Kr-na-mukesh-shankar-jaikishan-and-shailendra-joota-japani-lyrics",
            "Vibrate":              "https://genius.com/Kr-na-vibrate-lyrics",
            "Hola Amigo":           "https://genius.com/Kr-na-hola-amigo-lyrics",
            "Kaha Tak":             "https://genius.com/Kr-na-kaha-tak-lyrics",
            "Khatta Flow":          "https://genius.com/Seedhe-maut-and-kr-na-khatta-flow-lyrics",
        }
    },
    {
        "name": "Raftaar",
        "dir": "raftaar",
        "genius_id": 495985,
        "lang": "mixed",
        "tracks": {
            # Previously in corpus (lyricsmint-scraped, now restored via Genius)
            "36":                   "https://genius.com/Raftaar-36-lyrics",
            "Aage Chal":            "https://genius.com/Raftaar-aage-chal-lyrics",
            "Abbu":                 "https://genius.com/Raftaar-abbu-lyrics",
            "Aisa Main Shaitaan":   "https://genius.com/Raftaar-aisa-main-shaitaan-lyrics",
            "Badnaam":              "https://genius.com/Raftaar-badnaam-lyrics",
            "Barbaad":              "https://genius.com/Raftaar-barbaad-lyrics",
            "Beshaq":               "https://genius.com/Raftaar-beshaq-lyrics",
            "Black Sheep":          "https://genius.com/Raftaar-black-sheep-lyrics",
            "Chora Baba Ka":        "https://genius.com/Raftaar-chora-baba-ka-lyrics",
            "Damn":                 "https://genius.com/Raftaar-damn-lyrics",
            "Dehshat Ho":           "https://genius.com/Raftaar-dehshat-ho-lyrics",
            "Dhaakad":              "https://genius.com/Raftaar-dhaakad-lyrics",
            "Dilli Waali Baatcheet":"https://genius.com/Raftaar-dilli-waali-baatcheet-lyrics",
            "Do Hazaar Solo":       "https://genius.com/Raftaar-do-hazaar-solo-lyrics",
            "Down":                 "https://genius.com/Raftaar-down-lyrics",
            "Drama":                "https://genius.com/Raftaar-drama-lyrics",
            "F16":                  "https://genius.com/Raftaar-f16-lyrics",
            "Feeling You":          "https://genius.com/Raftaar-feeling-you-lyrics",
            "Gaddi":                "https://genius.com/Raftaar-gaddi-lyrics",
            "Gall Goriye":          "https://genius.com/Raftaar-gall-goriye-lyrics",
            "Gangnum":              "https://genius.com/Raftaar-gangnum-lyrics",
            "Ghana Kasoota":        "https://genius.com/Raftaar-ghana-kasoota-lyrics",
            "Go Pagal":             "https://genius.com/Raftaar-go-pagal-lyrics",
            "Goat Dekho":           "https://genius.com/Raftaar-goat-dekho-lyrics",
            "Haan":                 "https://genius.com/Raftaar-haan-lyrics",
            "Haseeno Ka Deewana":   "https://genius.com/Raftaar-haseeno-ka-deewana-lyrics",
            "Ice":                  "https://genius.com/Raftaar-ice-lyrics",
            "Jean Teri":            "https://genius.com/Raftaar-jean-teri-lyrics",
            "Kaali Car":            "https://genius.com/Raftaar-kaali-car-lyrics",
            "Kartootein":           "https://genius.com/Raftaar-kartootein-lyrics",
            "Load Hai":             "https://genius.com/Raftaar-load-hai-lyrics",
            "Lonely":               "https://genius.com/Raftaar-lonely-lyrics",
            "Main Wahi Hoon":       "https://genius.com/Raftaar-main-wahi-hoon-lyrics",
            "Mask On":              "https://genius.com/Raftaar-mask-on-lyrics",
            "Me And My Broski":     "https://genius.com/Raftaar-me-and-my-broski-lyrics",
            "Me And My Pen":        "https://genius.com/Raftaar-me-and-my-pen-lyrics",
            "Mera Parichay":        "https://genius.com/Raftaar-mera-parichay-lyrics",
            "Morni":                "https://genius.com/Raftaar-morni-lyrics",
            "Move":                 "https://genius.com/Raftaar-move-lyrics",
            "Munde Hood De":        "https://genius.com/Raftaar-munde-hood-de-lyrics",
            "Never Back Down":      "https://genius.com/Raftaar-never-back-down-lyrics",
            "No China":             "https://genius.com/Raftaar-no-china-lyrics",
            "Phone Mila Ke":        "https://genius.com/Raftaar-phone-mila-ke-lyrics",
            "Popular":              "https://genius.com/Raftaar-popular-lyrics",
            "Proud":                "https://genius.com/Raftaar-proud-lyrics",
            "Raashah":              "https://genius.com/Raftaar-raashah-lyrics",
            "Rap Ta":               "https://genius.com/Raftaar-rap-ta-lyrics",
            "Real Shit":            "https://genius.com/Raftaar-real-shit-lyrics",
            "Sick":                 "https://genius.com/Raftaar-sick-lyrics",
            "Superman":             "https://genius.com/Raftaar-superman-lyrics",
            "Tu Phir Se Aana":      "https://genius.com/Raftaar-tu-phir-se-aana-lyrics",
            "Warna Gabbar Aa Jayega":"https://genius.com/Raftaar-warna-gabbar-aa-jayega-lyrics",
            "Woh Raat":             "https://genius.com/Raftaar-woh-raat-lyrics",
            # New Genius additions
            "Swag Mera Desi":       "https://genius.com/Raftaar-swag-mera-desi-lyrics",
            "Sheikh Chilli":        "https://genius.com/Raftaar-sheikh-chilli-lyrics",
            "Mantoiyat":            "https://genius.com/Raftaar-and-nawazuddin-siddiqui-mantoiyat-lyrics",
            "Awein Hai":            "https://genius.com/Raftaar-awein-hai-lyrics",
            "Banjo Bounce":         "https://genius.com/Raftaar-and-epr-iyer-banjo-bounce-lyrics",
            "Naachne Ka Shaunq":    "https://genius.com/Raftaar-naachne-ka-shaunq-lyrics",
            "Trap Praa":            "https://genius.com/Raftaar-and-prabh-deep-trap-praa-lyrics",
            "Baawe":                "https://genius.com/Raftaar-and-badshah-baawe-lyrics",
        }
    },
]

def title_slug(title: str) -> str:
    t = title.lower()
    t = re.sub(r"\(.*?\)", "", t)
    t = t.replace("&", "and").replace("$", "s")
    t = re.sub(r"[^a-z0-9\s-]", "", t)
    t = re.sub(r"[\s-]+", "-", t).strip("-")
    return t

def search_genius_url_via_api(artist: str, track: str) -> str | None:
    import requests
    if not GENIUS_TOKEN:
        print("      x GENIUS_ACCESS_TOKEN not set in environment.")
        return None
    url = "https://api.genius.com/search"
    headers = {"Authorization": f"Bearer {GENIUS_TOKEN}"}
    params = {"q": f"{artist} {track}"}
    
    max_retries = 4
    sleep_time = 3.0
    for attempt in range(max_retries + 1):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            if r.status_code == 200:
                hits = r.json().get("response", {}).get("hits", [])
                
                # Normalize artist and create aliases
                target_lower = artist.lower()
                aliases = [target_lower]
                if "lil bhatia" in target_lower:
                    aliases.extend(["lil bhavi", "bhavi", "bhatia"])
                if "epr" in target_lower:
                    aliases.extend(["epr iyer", "epr"])
                if "panther" in target_lower:
                    aliases.extend(["panther (ind)", "panther"])
                    
                for hit in hits:
                    result = hit["result"]
                    primary_artist = result.get("primary_artist", {}).get("name", "").lower()
                    featured_artists = [f.get("name", "").lower() for f in result.get("featured_artists", [])]
                    
                    # Check for alias match
                    match_found = False
                    for alias in aliases:
                        if alias in primary_artist:
                            match_found = True
                            break
                        for fa in featured_artists:
                            if alias in fa:
                                match_found = True
                                break
                                
                    if match_found:
                        return result["url"]
                print(f"      x No matching artist found in Genius hits for query: '{artist} {track}'")
                break
            elif r.status_code == 429:
                print(f"      ! Rate limited (429) on '{artist} {track}', retrying in {sleep_time}s... (attempt {attempt + 1}/{max_retries + 1})")
                time.sleep(sleep_time)
                sleep_time *= 2.0
                continue
            else:
                print(f"      x Genius API returned status {r.status_code}")
                break
        except Exception as e:
            print(f"      x Genius API search exception: {e}")
            break
    return None

def clean_parsed_lyrics(lyrics: str) -> str:
    # Remove lines like "[Verse 1]", "[Chorus]", etc.
    lines = lyrics.splitlines()
    cleaned = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        if l.startswith("[") and l.endswith("]"):
            continue
        cleaned.append(l)
    
    res = "\n".join(cleaned).strip()
    # Remove any leftover "X Contributors Lyrics" at the start
    res = re.sub(r'^\d+\s+Contributors.*Lyrics\b', '', res, flags=re.IGNORECASE).strip()
    res = re.sub(r'^.*Lyrics\b', '', res, count=1, flags=re.IGNORECASE).strip()
    return res

def fetch_html_with_curl(url: str) -> str | None:
    import subprocess
    import tempfile
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as temp_file:
            temp_path = temp_file.name
        cmd = [
            "curl.exe",
            "-s",
            "-o", temp_path,
            "-H", "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            url
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            print(f"      x curl failed with code {res.returncode}")
            return None
        p = Path(temp_path)
        if p.exists():
            content = p.read_text(encoding="utf-8", errors="ignore")
            p.unlink()
            return content
        return None
    except Exception as e:
        print(f"      x Exception running curl: {e}")
        return None

async def scrape_real_lyrics(client: Mistral, artist_info: dict, track: str, url: str) -> str | None:
    print(f"    - Fetching: {url}")
    
    try:
        html = fetch_html_with_curl(url)
        if not html or len(html) < 2000:
            print("      x Failed to fetch HTML content.")
            return None
            
        soup = BeautifulSoup(html, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        
        containers = soup.find_all("div", attrs={"data-lyrics-container": "true"})
        if containers:
            text_parts = []
            for container in containers:
                for br in container.find_all("br"):
                    br.replace_with("\n")
                text_parts.append(container.get_text())
            raw_lyrics = "\n\n".join(text_parts).strip()
        else:
            old_container = soup.find("div", class_="lyrics")
            if old_container:
                raw_lyrics = old_container.get_text().strip()
            else:
                print("      x Could not find lyrics container in HTML.")
                return None
        cleaned = clean_parsed_lyrics(raw_lyrics)
        if len(cleaned) < 100:
            print(f"      x Scraped lyrics too short ({len(cleaned)} chars).")
            return None
            
        return cleaned
            
    except Exception as e:
        print(f"      x Exception during scraping: {e}")
        return None

async def main():
    if not API_KEY:
        print("ERROR: MISTRAL_API_KEY is not set.")
        return 1

    client = Mistral(api_key=API_KEY)
    
    for artist in TARGET_ARTISTS:
        print(f"\n=== Scraping REAL lyrics for {artist['name']} ===")
        out_dir = DATA_DIR / artist["dir"]
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect valid slugs so we can purge stale hallucinated files after
        valid_slugs = set()
        
        for track, url in artist["tracks"].items():
            slug = title_slug(track)
            valid_slugs.add(f"{slug}.json")
            path = out_dir / f"{slug}.json"
            
            # Skip if we already have a good-sized real file
            if path.exists() and path.stat().st_size > 1000:
                print(f"  - '{track}' already exists, skipping.")
                continue
            
            print(f"  - Scrape '{track}'...")
            real_lyrics = await scrape_real_lyrics(client, artist, track, url)
            
            if real_lyrics:
                rec = {
                    "artist": artist["name"],
                    "genius_artist_id": artist["genius_id"],
                    "title": track,
                    "lyricsmint_url": url,
                    "primary_language": artist["lang"],
                    "lyrics": real_lyrics,
                    "line_count": sum(1 for l in real_lyrics.splitlines() if l.strip()),
                    "char_count": len(real_lyrics),
                }
                path.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"    ✓ saved to {path.name} ({rec['line_count']} lines)")
            else:
                print(f"    ! failed to scrape '{track}' — skipping")
                
            time.sleep(1.5)  # politeness delay
        
        # NOTE: We do NOT purge files here — existing files may be real
        # songs scraped from lyricsmint or other sources that are not in
        # our current TARGET_ARTISTS track list. Only add, never delete.

    print("\nScraping of real lyrics finished!")
    return 0

if __name__ == "__main__":
    import asyncio
    raise SystemExit(asyncio.run(main()))

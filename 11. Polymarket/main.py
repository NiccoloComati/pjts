from py_clob_client.client import ClobClient

host = "https://clob.polymarket.com"
key = "0x5fa1a46c571bf12bc6a69a7cad3895d74492e81ab0a70ba7031579ef1b3874f3"  # Replace with your Polygon private key
chain_id = 137

client = ClobClient(host, key=key, chain_id=chain_id)
print(client.get_address())  # Should print your Polygon wallet address

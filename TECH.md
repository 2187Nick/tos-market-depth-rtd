# Dive deeper into level 2 data

![](https://github.com/user-attachments/assets/9a82789f-4fbe-4906-b34d-0823608c6fc4)

Through RTD(or DDE) I am only able to get data from some exchanges.

Example for NYSE Exchange Volume:

```=DDE("TOS","VOLUME",".SPY250417C555&N")```

You can see the Total Volume(Red box) and Total Exchange Volume(Green box)

### Why doesnt the Total exchange volume == Total volume?

- Some trades are transacted outside of the exchanges listed? 
- Why doesnt TOS list them?

### Available Exchanges
The exchanges available through RTD correspond directly to what's visible in the Level 2 data field within ThinkorSwim:
![](https://github.com/user-attachments/assets/0a0a4346-5948-486d-b5ed-7067526772db)



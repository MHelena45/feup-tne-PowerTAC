package org.powertac.samplebroker.util;

import org.json.JSONObject;
import org.springframework.stereotype.Service;

import java.util.HashMap;
import java.util.Map;
import static java.util.Map.entry;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.net.http.HttpRequest.BodyPublishers;
import java.net.http.HttpResponse.BodyHandlers;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.net.URI;
import java.net.http.HttpClient;

@Service("pyComs")
public class PyComs {
    // Context Manager
    public static HashMap<Integer, JSONObject> energyReportMap = new HashMap<>();
    public static HashMap<Integer, JSONObject> competitionJson = new HashMap<>();

    // Market Manager Service
    public static HashMap<Integer, ArrayList<JSONObject>> clearedTradeJson = new HashMap<>();
    public static HashMap<Integer, ArrayList<JSONObject>> orderbookJson = new HashMap<>();
    public static HashMap<Integer, JSONObject> weatherForecastJson = new HashMap<>();
    public static HashMap<Integer, JSONObject> weatherJson = new HashMap<>();
    public static int lastIndexCleared,  lastIndexOrderBook = 0;
    public static int mapLastIndexCleared = 360;
    public static int mapLastIndexOrderBook = 360;
    public static int numberOfCompetitors;
    public static int numberOfCustomers;

    public static Map<String, String> jsonType = Map.ofEntries(
        entry("energyReportType", "energyReportType"),
        entry("competitionJsonType", "competitionJsonType"),
        entry("clearedTradeJsonType", "clearedTradeJsonType"),
        entry("orderbookJsonType", "orderbookJsonType"),
        entry("weatherForecastJsonType", "weatherForecastJsonType"),
        entry("weatherJsonType", "weatherJsonType")
    );

    public static void trySend(Integer timeslot){
        var listOfMaps = new ArrayList<HashMap<Integer, JSONObject>>();
        listOfMaps.add(energyReportMap);
        listOfMaps.add(weatherForecastJson);
        listOfMaps.add(weatherJson);

        var listOfMaps2 = new ArrayList<HashMap<Integer, ArrayList<JSONObject>>>();
        listOfMaps2.add(clearedTradeJson);
        //listOfMaps2.add(orderbookJson);   //does not need to be send

        var shouldSend = true;

        for (HashMap<Integer,JSONObject> hashMap : listOfMaps) {
            if(hashMap.get(timeslot) == null){
                System.out.println("Failed");
                shouldSend = false;
                break;
            }
        }
        
        for (HashMap<Integer,ArrayList<JSONObject>> hashMap : listOfMaps2) {
            if(hashMap.get(timeslot) == null){
                System.out.println("Failed2");
                System.out.println(hashMap.toString());
                shouldSend = false;
                break;
            }
        }

        if(shouldSend){
            var toSend = new JSONObject();
            var value = new ArrayList<JSONObject>();
            var value2 = new ArrayList<ArrayList<JSONObject>>();

            for (HashMap<Integer, JSONObject> map : listOfMaps)
                value.add(map.get(timeslot));

            for (HashMap<Integer, ArrayList<JSONObject>> map : listOfMaps2)
                value2.add(map.get(timeslot));

            toSend.put("SampleType", "Training");
            toSend.put("noCompetitors", numberOfCompetitors);
            toSend.put("noCustomers", numberOfCustomers);
            toSend.put("SingleObjects", value);
            toSend.put("ListObjects", value2);

            var client = HttpClient.newHttpClient();
        
            File myObj = new File("file.json");

            try {
                if (! myObj.createNewFile()){
                    myObj.delete();
                    myObj.createNewFile();
                }
                
                FileWriter writer = new FileWriter("file.json");
                writer.write(toSend.toString());
                writer.close();

                var request = HttpRequest.newBuilder(
                    URI.create("http://localhost:4443"))
                    .header("Content-Type", "application/json")
                    .POST(BodyPublishers.ofFile(Paths.get("file.json")))
                .build();
                
                try {
                    HttpResponse<String> response = client.send(request, BodyHandlers.ofString());
                    System.out.println(response.body());
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } catch (IOException e1) {
                e1.printStackTrace();
            }
            finally {
                myObj.delete();
            }
        }
    }

    public JSONObject createMockClearedTrade(int slotInDay){
        var clearedTrade = new HashMap<String, Object>();
        clearedTrade.put("timeslotIndex", "none");
        clearedTrade.put("executionMWh", "none");
        clearedTrade.put("executionPrice", "none");
        clearedTrade.put("dateExecuted", "none");
        clearedTrade.put("slotInDay", slotInDay);
        return new JSONObject(clearedTrade);
    }

    public JSONObject createMockOrderbook(int slotInDay){
        var orderbook = new HashMap<String, Object>();
        orderbook.put("timeslotIndex", "none");
        orderbook.put("clearingPrice", "none");
        orderbook.put("asks", "none");
        orderbook.put("bids", "none");
        orderbook.put("slotInDay", slotInDay);
        return new JSONObject(orderbook);
    }

    public void trigger(JSONObject obj, String type){
        String currSlot = obj.get("timeslotIndex").toString(); 
        
        switch(type){
            case "energyReportType":
                energyReportMap.put(Integer.parseInt(currSlot), obj);
                //Fills missing values if trades on 0 slot did not happen
                if(lastIndexCleared == 23) {
                    ArrayList<JSONObject> clearedTrades;
                    clearedTrades =  clearedTradeJson.get(mapLastIndexCleared); 
                    clearedTrades.add(createMockClearedTrade(0)); //0 is the last index
                    lastIndexCleared = 0;
                }
                if(lastIndexOrderBook == 23) {
                    ArrayList<JSONObject> orderBooks;                                   
                    orderBooks =  orderbookJson.get(mapLastIndexOrderBook); 
                    orderBooks.add(createMockOrderbook(0)); //0 is the last index
                    lastIndexOrderBook = 0;
                }
                trySend(Integer.parseInt(currSlot));
                break;
            case "competitionJsonType":
                competitionJson.put(Integer.parseInt(currSlot), obj);
                numberOfCompetitors = (Integer) obj.get("noCompetitors");
                numberOfCustomers = (Integer) obj.get("noCustomer");
                break;
            case "clearedTradeJsonType":
                int slotInDay = Integer.parseInt(obj.get("slotInDay").toString()); 

                if(lastIndexCleared - 1 == slotInDay) {
                    System.out.println("Message was late");
                } else if(lastIndexCleared - 2 == slotInDay) {
                    System.out.println("Message was late");
                } else {
                    //Fills missing values
                    ArrayList<JSONObject> clearedTrades;
                    //when slotInDay missing is 0, it is fix in energyReportType case
                    if(slotInDay == 0 && lastIndexCleared == 22) {
                        clearedTrades =  clearedTradeJson.get(mapLastIndexCleared); 
                        clearedTrades.add(createMockClearedTrade(23));
                        lastIndexCleared = 0;
                    } else if(slotInDay == 1 || (slotInDay == 2 && lastIndexCleared == 0)  || (slotInDay == 3 && lastIndexCleared == 0) || (slotInDay == 4 && lastIndexCleared == 0)) {
                        clearedTrades =  clearedTradeJson.get(mapLastIndexCleared);  
                        clearedTrades = new ArrayList<>();
                        mapLastIndexCleared++;
                        for(int i=lastIndexCleared+1; i < slotInDay; i++) {
                            clearedTrades.add(createMockClearedTrade(i));
                        } 
                    } else {
                        clearedTrades =  clearedTradeJson.get(mapLastIndexCleared); 
                        for(int i=lastIndexCleared +1; i < slotInDay; i++) {
                            clearedTrades.add(createMockClearedTrade(i));
                        } 
                    }  
                    clearedTrades.add(obj);
                    clearedTradeJson.put(mapLastIndexCleared, clearedTrades);  
                    lastIndexCleared = slotInDay;
                }
                break;
            case "orderbookJsonType":
                int slotInDayOrder = Integer.parseInt(obj.get("slotInDay").toString()); 

                if(lastIndexOrderBook - 1 == slotInDayOrder) {
                    System.out.println("Message was late");
                } else if(lastIndexOrderBook - 2 == slotInDayOrder) {
                    System.out.println("Message was late");
                } else {
                    //Fills missing values
                    ArrayList<JSONObject> orderBooks;
                    //when slotInDay missing is 0, it is fix in energyReportType case
                    if(slotInDayOrder == 0 && lastIndexOrderBook == 22) {
                        orderBooks =  orderbookJson.get(mapLastIndexOrderBook); 
                        orderBooks.add(createMockOrderbook(23));
                        lastIndexOrderBook = 0;
                    } else if(slotInDayOrder == 1 || (slotInDayOrder == 2 && lastIndexOrderBook == 0)  || (slotInDayOrder == 3 && lastIndexOrderBook == 0) || (slotInDayOrder == 4 && lastIndexOrderBook == 0)) {           
                        orderBooks =  orderbookJson.get(mapLastIndexOrderBook);  
                        orderBooks = new ArrayList<>();
                        mapLastIndexOrderBook++;
                        for(int i=lastIndexOrderBook+1; i < slotInDayOrder; i++) {
                            orderBooks.add(createMockOrderbook(i));
                        } 
                    } else {
                        orderBooks =  orderbookJson.get(mapLastIndexOrderBook); 
                        for(int i=lastIndexOrderBook +1; i < slotInDayOrder; i++) {
                            orderBooks.add(createMockOrderbook(i));
                        } 
                    }  
                    orderBooks.add(obj);
                    orderbookJson.put(mapLastIndexOrderBook, orderBooks);  
                    lastIndexOrderBook = slotInDayOrder;
                }
                break;
            case "weatherForecastJsonType":
                weatherForecastJson.put(Integer.parseInt(currSlot), obj);
                break;
            case "weatherJsonType":
                weatherJson.put(Integer.parseInt(currSlot), obj);
                break;
        }
    }
}

import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Papa from 'papaparse';

const WeeklySleepPatterns = () => {
  const [weeklyData, setWeeklyData] = useState([]);
  const [selectedWeek, setSelectedWeek] = useState(0);
  const [hourlyData, setHourlyData] = useState([]);
  
  useEffect(() => {
    const analyzeData = async () => {
      const fileContent = await window.fs.readFile('categorized_output.csv', { encoding: 'utf8' });
      const parsedData = Papa.parse(fileContent, {
        header: true,
        skipEmptyLines: true
      });
      
      // Convert timestamps and sort
      const activities = parsedData.data
        .map(row => ({
          ...row,
          timestamp: new Date(row.timestamp)
        }))
        .sort((a, b) => a.timestamp - b.timestamp);

      // Generate weeks
      const weeks = [];
      let currentStart = new Date(activities[0].timestamp);
      currentStart.setDate(currentStart.getDate() - currentStart.getDay());
      const endDate = activities[activities.length - 1].timestamp;
      
      while (currentStart < endDate) {
        const weekEnd = new Date(currentStart);
        weekEnd.setDate(weekEnd.getDate() + 7);
        
        const weekActivities = activities.filter(
          act => act.timestamp >= currentStart && act.timestamp < weekEnd
        );
        
        if (weekActivities.length > 0) {
          const hourlyInactivity = Array(24).fill(0);
          const inactiveDurations = Array(24).fill().map(() => []);
          
          // Calculate inactive hours
          for (let day = 0; day < 7; day++) {
            const dayStart = new Date(currentStart);
            dayStart.setDate(dayStart.getDate() + day);
            
            for (let hour = 0; hour < 24; hour++) {
              const hourStart = new Date(dayStart);
              hourStart.setHours(hour, 0, 0, 0);
              const hourEnd = new Date(dayStart);
              hourEnd.setHours(hour, 59, 59, 999);
              
              const hasActivity = weekActivities.some(
                act => act.timestamp >= hourStart && act.timestamp <= hourEnd
              );
              
              if (!hasActivity) {
                hourlyInactivity[hour]++;
              }
            }
          }
          
          // Calculate inactive durations
          for (let i = 0; i < weekActivities.length - 1; i++) {
            const gap = (weekActivities[i + 1].timestamp - weekActivities[i].timestamp) / (1000 * 60 * 60);
            if (gap >= 2) {
              const startHour = weekActivities[i].timestamp.getHours();
              inactiveDurations[startHour].push(gap);
            }
          }
          
          weeks.push({
            startDate: currentStart.toISOString().split('T')[0],
            endDate: weekEnd.toISOString().split('T')[0],
            inactivePercentages: hourlyInactivity.map(count => (count / 7) * 100),
            avgDurations: inactiveDurations.map(durations => 
              durations.length > 0 
                ? durations.reduce((a, b) => a + b) / durations.length 
                : 0
            )
          });
        }
        
        currentStart = weekEnd;
      }
      
      setWeeklyData(weeks);
      updateHourlyData(weeks[0]);
    };
    
    analyzeData();
  }, []);
  
  const updateHourlyData = (weekData) => {
    const data = Array(24).fill(0).map((_, hour) => ({
      hour: `${hour.toString().padStart(2, '0')}:00`,
      inactivePercentage: weekData.inactivePercentages[hour],
      avgDuration: weekData.avgDurations[hour]
    }));
    setHourlyData(data);
  };

  const handleWeekChange = (index) => {
    setSelectedWeek(index);
    updateHourlyData(weeklyData[index]);
  };

  return (
    <Card className="w-full">
      <CardHeader className="space-y-4">
        <CardTitle>Weekly Sleep Pattern Analysis</CardTitle>
        <div className="flex flex-wrap gap-2">
          {weeklyData.map((week, index) => (
            <button
              key={week.startDate}
              onClick={() => handleWeekChange(index)}
              className={`px-3 py-1 text-sm rounded ${
                selectedWeek === index 
                  ? 'bg-blue-500 text-white' 
                  : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              {week.startDate}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent>
        <div className="mt-4 mb-2">
          <div className="flex justify-center gap-8">
            <div className="flex items-center">
              <div className="w-4 h-4 bg-[#8884d8] mr-2"></div>
              <span className="text-sm">% of Days Inactive</span>
            </div>
            <div className="flex items-center">
              <div className="w-4 h-4 bg-[#82ca9d] mr-2"></div>
              <span className="text-sm">Avg Duration When Inactive (hours)</span>
            </div>
          </div>
        </div>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={hourlyData}
              margin={{ top: 20, right: 80, left: 80, bottom: 40 }}
            >
              <XAxis
                dataKey="hour"
                label={{ value: 'Hour of Day', position: 'bottom', offset: 20 }}
              />
              <YAxis
                yAxisId="left"
                label={{ 
                  value: '% Days Inactive', 
                  angle: -90, 
                  position: 'insideLeft', 
                  offset: -60 
                }}
              />
              <YAxis
                yAxisId="right"
                orientation="right"
                label={{ 
                  value: 'Avg Hours When Inactive', 
                  angle: 90, 
                  position: 'insideRight',
                  offset: -50
                }}
              />
              <Tooltip />
              <Bar
                yAxisId="left"
                dataKey="inactivePercentage"
                fill="#8884d8"
              />
              <Bar
                yAxisId="right"
                dataKey="avgDuration"
                fill="#82ca9d"
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-8 space-y-2">
          <p className="text-sm text-gray-600">
            Showing data for week of {weeklyData[selectedWeek]?.startDate} to {weeklyData[selectedWeek]?.endDate}
          </p>
          <p className="text-sm text-gray-600">
            Blue bars show the percentage of days where each hour had no activity.
            Green bars show how long inactivity lasted when it started at that hour.
          </p>
        </div>
      </CardContent>
    </Card>
  );
};

export default WeeklySleepPatterns;
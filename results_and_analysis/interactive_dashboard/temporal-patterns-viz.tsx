import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import Papa from 'papaparse';

const TemporalPatternsViz = () => {
  const [data, setData] = useState({
    hourly: [],
    daily: [],
    weekly: []
  });

  useEffect(() => {
    const processData = async () => {
      const fileContent = await window.fs.readFile('categorized_output.csv', { encoding: 'utf8' });
      const parsedData = Papa.parse(fileContent, {
        header: true,
        skipEmptyLines: true
      });
      
      // Convert timestamps
      const activities = parsedData.data.map(row => ({
        ...row,
        timestamp: new Date(row.timestamp.replace('EST', '').trim())
      }));

      // Hourly distribution
      const hourlyDist = Array(24).fill(0);
      activities.forEach(act => {
        const hour = act.timestamp.getHours();
        hourlyDist[hour]++;
      });
      const hourlyData = hourlyDist.map((count, hour) => ({
        hour: `${hour.toString().padStart(2, '0')}:00`,
        count
      }));

      // Daily distribution
      const dailyDist = {};
      activities.forEach(act => {
        const date = act.timestamp.toISOString().split('T')[0];
        dailyDist[date] = (dailyDist[date] || 0) + 1;
      });
      const dailyData = Object.entries(dailyDist).map(([date, count]) => ({
        date,
        count
      })).sort((a, b) => a.date.localeCompare(b.date));

      // Weekly distribution
      const weeklyDist = {};
      activities.forEach(act => {
        const day = act.timestamp.toLocaleDateString('en-US', { weekday: 'long' });
        weeklyDist[day] = (weeklyDist[day] || 0) + 1;
      });
      const daysOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
      const weeklyData = daysOrder.map(day => ({
        day,
        count: weeklyDist[day] || 0
      }));

      setData({
        hourly: hourlyData,
        daily: dailyData,
        weekly: weeklyData
      });
    };

    processData();
  }, []);

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle>Hourly Activity Distribution</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.hourly}>
                <XAxis dataKey="hour" angle={-45} textAnchor="end" height={60} />
                <YAxis label={{ value: 'Number of Activities', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#8884d8" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-gray-600">
            Shows the distribution of activities across different hours of the day
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Daily Activity Trend</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={data.daily}>
                <XAxis 
                  dataKey="date" 
                  angle={-45} 
                  textAnchor="end" 
                  height={60}
                  interval="preserveStartEnd"
                />
                <YAxis label={{ value: 'Number of Activities', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="count" stroke="#82ca9d" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-gray-600">
            Shows how activity levels change over days
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Weekly Activity Pattern</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data.weekly}>
                <XAxis dataKey="day" />
                <YAxis label={{ value: 'Number of Activities', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Bar dataKey="count" fill="#82ca9d" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-4 text-sm text-gray-600">
            Shows activity distribution across days of the week
          </p>
        </CardContent>
      </Card>
    </div>
  );
};

export default TemporalPatternsViz;
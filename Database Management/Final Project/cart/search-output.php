<table>
<tr><th>商品id</th><th>商品名稱</th><th>商品價格</th><th>目前存貨數</th><th>簡介</th><th>圖片</th></tr>
<?PHP
$pdo=new PDO('mysql:host=localhost;dbname=final;charset=utf8','root','');    //連結資料庫
$sql=$pdo->prepare('select * from commodity where c_name like ?');    //執行Query查詢 
$sql->execute(['%'.$_REQUEST['keyword'].'%']);
foreach($sql->fetchAll() as $row){
    echo '<tr>';
    echo '<td>',$row['c_id'],'</td>';
    echo '<td>',$row['c_name'],'</td>';
    echo '<td>',$row['price'],'</td>';
    echo '<td>',$row['quantity'],'</td>';
    echo '<td>',$row['describe'],'</td>';
    echo '<td><img src="',$row['url1'],'" width=600px></td>';
}
?>